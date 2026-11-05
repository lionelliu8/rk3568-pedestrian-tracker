set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-11)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-11)

set(RK3568_SYSROOT "/root/npu/ubuntu22-arm64" CACHE PATH "ARM64 Ubuntu sysroot")
set(CMAKE_FIND_ROOT_PATH "${RK3568_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,-rpath-link,${RK3568_SYSROOT}/lib/aarch64-linux-gnu -Wl,-rpath-link,${RK3568_SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${RK3568_SYSROOT}/usr/lib -Wl,-rpath-link,${RK3568_SYSROOT}/usr/lib/aarch64-linux-gnu/blas -Wl,-rpath-link,${RK3568_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack")
