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
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

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

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
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
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;
    m_params.isColliding = false;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

    m_params.breakingTension = 10.0f;
    m_params.ropeSpring = 8.0f;
    m_params.minOscillations = 50;
    m_params.tCycle = 6000.0f; // 120.0f * m_params.minOscillations;

    _initialize(numParticles);
//    reset(CONFIG_PEND);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    printf("createVBO\n");
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}
/*
uint
ParticleSystem::createLenVBO(uint size, )
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}
*/
inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
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

    if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
        m_lenVbo = createVBO(memSize);
        printf("On initialize(). m_lenVbo created, registeringGLBufferObject\n");
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
        registerGLBufferObject(m_lenVbo, &m_cuda_lenvbo_resource);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
        checkCudaErrors(cudaMalloc((void **)&m_cudaLenVBO, memSize)) ;
        printf("On initialize(). Not using OpenGL, allocated m_cudaLenVBO\n");
    }

    allocateArray((void **)&m_dVel, memSize);
    //allocateArray((void **)&m_dLen, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
    allocateArray((void **)&m_dSortedLen, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    if (m_bUseOpenGL)
    {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffers
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

        for (uint i=0; i<m_numParticles; i++)
        {
            float t = i / (float) m_numParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
            colorRamp(t, ptr);
            ptr+=3;
#endif
            *ptr++ = 1.0f;
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);

    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

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
    //freeArray(m_dLen);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
    freeArray(m_dSortedLen);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_colorvbo_resource);
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        unregisterGLBufferObject(m_cuda_lenvbo_resource);
        printf("On finalize() unregisterGLBufferObject m_cuda_lenvbo_resource\n");
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
        glDeleteBuffers(1, (const GLuint *)&m_lenVbo);
        printf("On finalize() deletedBuffers m_lenVbo\n");
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
        checkCudaErrors(cudaFree(m_cudaLenVBO));
        printf("On finalize(). Not using OpenGL, freed m_cudaLenVBO\n");
    }
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;
    float *dLen;

    if (m_bUseOpenGL)
    {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
        dLen = (float *) mapGLBufferObject(&m_cuda_lenvbo_resource);
        printf("update() started. MappedGLBufferObject m_cuda_lenvbo_resource\n");
    }
    else
    {
        dPos = (float *) m_cudaPosVBO;
        dLen = (float *) m_cudaLenVBO;
        printf("update() started. Not using OpenGL, uses m_cudaLenVBO\n");
    }

    // update constants
    setParameters(&m_params);

    // integrate
    integrateSystem(
        dPos,
        m_dVel,
	dLen,
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
	dLen,
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

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
        unmapGLBufferObject(m_cuda_lenvbo_resource);
        printf("update() finishing. UnmappedGLBufferObject m_cuda_lenvbo_resource\n");
    }
}

void
ParticleSystem::dumpGrid()
{
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
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
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
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            printf("getArray for position\n");
            hdata = m_hPos;
            ddata = m_dPos;
            cuda_vbo_resource = m_cuda_posvbo_resource;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;

	case LENGTH:
            printf("getArray for length\n");
	    hdata = m_hLen;
	    ddata = m_dVel;
            cuda_vbo_resource = m_cuda_lenvbo_resource;
	    break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
//  for empty vbo it's == checkCudaErrors(cudaMemcpy(hdata, ddata, m_numParticles*4*sizeof(float), cudaMemcpyDeviceToHost));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    printf("setArray for position\n");
                    unregisterGLBufferObject(m_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
                }
		//else
		//    copyArrayToDevice(m_dPos, data, start*4*sizeof(float), count*4*sizeof(float));
            }
            break;

        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
	    
	case LENGTH:
            {
                if (m_bUseOpenGL)
                {
                    printf("setArray for length\n");
                    unregisterGLBufferObject(m_cuda_lenvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_lenVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_lenVbo, &m_cuda_lenvbo_resource);
                }
		else
		    copyArrayToDevice(m_cudaLenVBO, data, start*4*sizeof(float), count*4*sizeof(float));
            }
            //copyArrayToDevice(m_dLen, data, start*4*sizeof(float), count*4*sizeof(float));
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
                printf("reset() CONFIG_PEND\n");
		float initOffset = m_params.particleRadius * 1.0f;
		float startx = initOffset - 1.0f;
		float spacing = m_params.particleRadius * 3.0f;
	        uint maxParticles = floorf((2.0f - 2.0f * initOffset) / spacing) + 1.0f;
	        if(m_numParticles > maxParticles)
	        {
		    printf("max particles number exceeded, adopting max possible value = %d, provided value: %d\n", maxParticles, m_numParticles);
		    m_numParticles = maxParticles;
	        }

                int p = 0, v = 0, l = 0;
		float pisq = powf(CUDART_PI_F, 2.0f);
		float tsq = powf(m_params.tCycle, 2.0f);
	        float g = -m_params.gravity.y;
		float len = (g * tsq) / (4.0f * pisq * powf(m_params.minOscillations + m_numParticles - 1, 2.0f));
		float maxDisplacement = len / 6.0f; // small angle approximation works for angles <= 1/9 rad
		float maxDisplacementSq = powf(maxDisplacement, 2.0f);

                for (uint i=0; i < m_numParticles; i++)
                {
		    len = (g * tsq) / (4.0f * pisq * powf(m_params.minOscillations + i, 2.0f));
		    m_hLen[l++] = startx;
		    m_hLen[l++] = 0.0f;
		    m_hLen[l++] = 0.0f;
		    m_hLen[l++] = len;// length
                    m_hPos[p++] = startx;
                    m_hPos[p++] = -powf(powf(len, 2.0f) - maxDisplacementSq, 1.0f / 2.0f);
                    m_hPos[p++] = maxDisplacement;
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
		    startx += spacing;
                }
	    }
	    break;

	case CONFIG_NEWTON:
	    {
                printf("reset() CONFIG_NEWTON\n");
		float len = 0.1f;
		float initOffset = len;
		float startx = initOffset - 1.0f;
		float spacing = m_params.particleRadius * 2.0f;
	        uint maxParticles = floorf((2.0f - 2.0f * initOffset) / spacing) + 1.0f;
	        if(m_numParticles > maxParticles)
	        {
		    printf("max particles number exceeded, adopting max possible value = %d, provided value: %d\n", maxParticles, m_numParticles);
		    m_numParticles = maxParticles;
	        }
                int p = 0, v = 0, l = 0;

		// The pendulum that starts the craddle
		m_hLen[l++] = startx;
		m_hLen[l++] = 0.0f;
		m_hLen[l++] = 0.0f;
		m_hLen[l++] = len; // length
	        m_hPos[p++] = startx;//len / 2.0f;
	        m_hPos[p++] = -(powf(3.0f, 1.0f / 2.0f) / 2.0f) * len;
	        m_hPos[p++] = 0.0f;
	        m_hPos[p++] = 1.0f; // radius
	        m_hVel[v++] = 0.0f;
	        m_hVel[v++] = 0.0f;
	        m_hVel[v++] = 0.0f;
	        m_hVel[v++] = 0.0f;

                for (uint i=1; i < m_numParticles; i++)
                {
		    startx += spacing;
		    m_hLen[l++] = startx;
		    m_hLen[l++] = 0.0f;
		    m_hLen[l++] = 0.0f;
		    m_hLen[l++] = len;// length
                    m_hPos[p++] = startx;
                    m_hPos[p++] = -len;
                    m_hPos[p++] = 0.0f;
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

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;

                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
                {
                    m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
