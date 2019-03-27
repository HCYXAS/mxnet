#pragma once

#ifdef __HIPCC__
    #define HOSTDEVICE __host__ __device__
#else
    #define HOSTDEVICE
#endif
