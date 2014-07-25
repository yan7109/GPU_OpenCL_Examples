#define iWindowSize 3
#define uiIterations 1
//*****************************************************************************
__kernel void ckAverage(__global uchar4* uc4Source, __global unsigned int* uiDest, __local uchar4* uc4LocalData, int iLocalPixPitch, int iImageWidth, int iDevImageHeight)//, int uiIterations)//, int iWindowSize)
{
    // Get parent image x and y pixel coordinates from global ID, and compute offset into parent GMEM data
    int halfWindow = iWindowSize/2;
    int padding = uiIterations - 1 + halfWindow;
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);
    int global_x = (local_size_x - 2*padding)*get_group_id(0) + local_x - padding; 
    int global_y = (local_size_y - 2*padding)*get_group_id(1) + local_y - padding; 
    unsigned int result = 0;

    int iDevGMEMOffset = mul24(global_y, iImageWidth) + global_x; 

    // Compute initial offset of current pixel within work group LMEM block
    int iLocalPixOffset = mul24(local_y, iLocalPixPitch) + local_x;

    // Main read of GMEM data into LMEM
    if((global_y > -1) && (global_y < iDevImageHeight) && (global_x > -1) && (global_x < iImageWidth))
    {
        uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset];
    }

    //Processing the four corner regions
    //Lower left
    if (global_x < 0 && global_y < 0)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[padding*iLocalPixPitch + padding];
    }
    else if (global_x >= iImageWidth && global_y < 0)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[(padding + 1)*iLocalPixPitch - padding - 1];
    }
    else if (global_x < 0 && global_y >= iDevImageHeight)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[(local_size_y - padding - 1)*iLocalPixPitch + padding];
    }
    else if (global_x >= iImageWidth && global_y >= iDevImageHeight)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[(local_size_y - padding)*iLocalPixPitch - padding - 1];
    }
    // Processing the four edges
    else if (global_y < 0)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[local_x + padding*iLocalPixPitch];
    }
    else if (global_y >= iDevImageHeight)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[local_x + (local_size_y - padding - 1)*iLocalPixPitch];
    }
    else if (global_x < 0)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[iLocalPixOffset + padding - local_x];
    }
    else if (global_x > iImageWidth)
    {
        uc4LocalData[iLocalPixOffset] = uc4LocalData[iLocalPixOffset - local_x + iLocalPixPitch - padding - 1];
    }

    if(local_x >= halfWindow && local_x < (local_size_x - halfWindow) && local_y >= halfWindow && local_y < (local_size_y - halfWindow))
    {

        for (int iter = 0; iter < uiIterations; ++iter) 
        {
            // Synchronize the read into LMEM
            barrier(CLK_LOCAL_MEM_FENCE);
            result = 0;
            unsigned int b_sum = 0;
            unsigned int g_sum = 0;
            unsigned int r_sum = 0;

            int index = iLocalPixOffset;
      
            for (int i = -1*halfWindow; i <= halfWindow; i+=1)
            {   
                for (int j = -1*halfWindow; j <= halfWindow; j+=1) 
                {
                    r_sum += (unsigned int)(uc4LocalData[index + i*iLocalPixPitch + j].x);
                    g_sum += (unsigned int)(uc4LocalData[index + i*iLocalPixPitch + j].y);
                    b_sum += (unsigned int)(uc4LocalData[index + i*iLocalPixPitch + j].z);
                }
            }   

            result = (r_sum/iWindowSize/iWindowSize & 0x000000FF) + (((g_sum/iWindowSize/iWindowSize)<<8) & 0x0000FF00) + (((b_sum/iWindowSize/iWindowSize)<<16) & 0x00FF0000);

            uchar4 input = (uchar4)0;
            input = (uchar4) result;
            input.x = result & 0xFF;
            input.y = (result >> 8) & 0xFF;
            input.z = (result >> 16) & 0xFF;
            input.w = (result >> 24) & 0x00;
            uc4LocalData[iLocalPixOffset] = input;
        }

        uiDest[iDevGMEMOffset] = result;
    }
}
