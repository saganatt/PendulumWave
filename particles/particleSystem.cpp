/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// OpenGL Graphics includes
//#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
//#include <helper_gl.h>

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, float tCycle, uint minOscillations) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_tCycle(tCycle),
    m_minOscillations(minOscillations),
    m_hPos(0),
    m_hVel(0),
    m_hLen(0),
    m_dPos(0),
    m_dVel(0),
    m_dLen(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    printf("ParticleSystem constructor\n");
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;
    m_params.breakingTension = 10.0f;

    m_params.gravity = make_float3(0.0f, 2.0f, 0.0f);
    m_params.globalDamping = 1.0f;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    printf("ParticleSystem destructor\n");
    _finalize();
    m_numParticles = 0;
}

void
ParticleSystem::_initialize(int numParticles)
{
    printf("ParticleSystem::_initialize\n");
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
    m_hLen = new float[m_numParticles * 4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
    memset(m_hLen, 0, m_numParticles*4*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;

    allocateArray((void **)&m_dVel, memSize);
    allocateArray((void **)&m_dLen, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
    allocateArray((void **)&m_dSortedLen, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));
    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hLen;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dVel);
    freeArray(m_dLen);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
    freeArray(m_dSortedLen);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    checkCudaErrors(cudaFree(m_cudaPosVBO));
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos = (float *) m_cudaPosVBO;

    // update constants
    setParameters(&m_params);

    // integrate
    integrateSystem(
        dPos,
        m_dVel,
	m_dLen,
        deltaTime,
        m_numParticles);

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
	m_dSortedLen,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_dVel,
	m_dLen,
        m_numParticles,
        m_numGridCells);

    // process collisions
    collide(
        m_dVel,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);
}

void
ParticleSystem::dumpGrid()
{
    printf("ParticleSystem::dumpGrid\n");
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    printf("ParticleSystem::dumpParticles\n");
    // debug
    copyArrayFromDevice(m_hPos, 0, 0/*&m_cuda_posvbo_resource*/, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

    for (uint i=start; i<start+count; i++)
    {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    printf("ParticleSystem::getArray\n");
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;

	case LENGTH:
	    hdata = m_hLen;
	    ddata = m_dVel;
	    break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
//  for empty vbo it's == checkCudaErrors(cudaMemcpy(hdata, ddata, m_numParticles*4*sizeof(float), cudaMemcpyDeviceToHost));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    printf("ParticleSystem::setArray\n");
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
	    copyArrayToDevice(m_cudaPosVBO, data, start*4*sizeof(float), count*4*sizeof(float));
            break;

        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
	    
	case LENGTH:
            copyArrayToDevice(m_dLen, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    printf("ParticleSystem::initGrid\n");
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+3] = 1.0f;

                    m_hVel[i*4] = 0.0f;
                    m_hVel[i*4+1] = 0.0f;
                    m_hVel[i*4+2] = 0.0f;
                    m_hVel[i*4+3] = 0.0f;

		    m_hLen[i*4] = -1.0f;
		    m_hLen[i*4+1] = -1.0f;
		    m_hLen[i*4+2] = -1.0f;
		    m_hLen[i*4+3] = -1.0f;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
    printf("ParticleSystem::reset\n");
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0, l = 0;

                for (uint i=0; i < m_numParticles; i++)
                {
                    float point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    m_hPos[p++] = 2 * (point[0] - 0.5f);
                    m_hPos[p++] = 2 * (point[1] - 0.5f);
                    m_hPos[p++] = 2 * (point[2] - 0.5f);
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
		    m_hLen[l++] = -1.0f;
		    m_hLen[l++] = -1.0f;
		    m_hLen[l++] = -1.0f;
		    m_hLen[l++] = -1.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                float jitter = m_params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
            }
            break;

	case CONFIG_PEND:
	    {
                int p = 0, v = 0, l = 0;
		float pisq = powf(CUDART_PI_F, 2.0f);
		float tsq = powf(m_tCycle, 2.0f);
	        float g = -m_params.gravity.y;
		float initzsq = powf(0.5f, 2.0f);
		float initOffset = m_params.particleRadius * 1.0f;
		float spacing = m_params.particleRadius * 3.0f;

                for (uint i=0; i < m_numParticles; i++)
                {
		    m_hLen[l++] = -1.0f + initOffset + (float)i * spacing;
		    m_hLen[l++] = 0.0f;
		    m_hLen[l++] = 0.0f;
		    m_hLen[l++] = (g * tsq) / (4.0f * pisq * powf(m_minOscillations + i, 2.0f)); // length
                    m_hPos[p++] = m_hLen[p - 1];
                    m_hPos[p++] = powf(powf(m_hLen[i], 2.0f) - initzsq, 1.0f / 2.0f);
                    m_hPos[p++] = 0.5f;
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
	    }
	    break;
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
    setArray(LENGTH, m_hLen, 0, m_numParticles);
}
