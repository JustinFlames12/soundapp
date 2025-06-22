#!/usr/bin/env bash
set -e

# 1) Path to your Android NDK (r21e+)
# NDK=${NDK:-/path/to/android-ndk-r21e}
NDK=/home/justin/android-ndk

# 2) Minimum Android API level
API_LEVEL=21

# 3) Where you cloned FFmpeg
FFMPEG_SRC_DIR=$(pwd)/ffmpeg-android

# 4) Where to install each build
OUTPUT_DIR=$(pwd)/android-binaries

# 5) Target ABIs
ABIS=("armeabi-v7a" "arm64-v8a")

# 6) Triples, arch & cpu maps
declare -A HOSTS=(
  ["armeabi-v7a"]="armv7a-linux-androideabi"
  ["arm64-v8a"]="aarch64-linux-android"
)
declare -A ARCHS=(
  ["armeabi-v7a"]="arm"
  ["arm64-v8a"]="aarch64"
)
declare -A CPUS=(
  ["armeabi-v7a"]="armv7-a"
  ["arm64-v8a"]="armv8-a"
)

for ABI in "${ABIS[@]}"; do
  echo "=== Building FFmpeg for $ABI ==="

  HOST=${HOSTS[$ABI]}
  ARCH=${ARCHS[$ABI]}
  CPU=${CPUS[$ABI]}

  TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64
  SYSROOT=$TOOLCHAIN/sysroot
  export PATH=$TOOLCHAIN/bin:$PATH       # ← add this line



  # point CC/CXX at the right wrappers:
  export CC=$TOOLCHAIN/bin/${HOST}${API_LEVEL}-clang
  export CXX=$TOOLCHAIN/bin/${HOST}${API_LEVEL}-clang++
  export AR=$TOOLCHAIN/bin/llvm-ar
  export LD=$TOOLCHAIN/bin/ld.lld
  export STRIP=$TOOLCHAIN/bin/llvm-strip
  export NM=$TOOLCHAIN/bin/llvm-nm
  export RANLIB=$TOOLCHAIN/bin/llvm-ranlib

  echo "### Building for $ABI ###"
  echo "CC = $CC"
  echo "SYSROOT = $SYSROOT"

  # Quick compile check
  echo 'int main(){}' > dummy.c
  $CC --sysroot=$SYSROOT -o dummy dummy.c \
    && echo "→ test compile OK" \
    || { cat dummy.c; echo "→ test compile FAIL"; exit 1; }

#   # now configure ffmpeg
#   ./configure \
#     --prefix=$PREFIX \
#     --target-os=android \
#     --arch=$ARCH \
#     --cpu=$CPU \
#     --enable-cross-compile \
#     --disable-shared \
#     --enable-static \
#     --sysroot=$SYSROOT \
#     --extra-cflags="-Os -fpic" \
#     --disable-debug \
#     --disable-doc \
#     --disable-ffplay \
#     --disable-ffprobe





  PREFIX=$OUTPUT_DIR/$ABI
  mkdir -p $PREFIX

  cd $FFMPEG_SRC_DIR
  make distclean >/dev/null 2>&1 || true

  # after you set STRIP:
  export INSTALL_STRIP=$STRIP





  ./configure \
    --prefix=$PREFIX \
    --target-os=android \
    --arch=$ARCH \
    --cpu=$CPU \
    --enable-cross-compile \
    --cc="$CC" \
    --cross-prefix="" \
    --sysroot=$SYSROOT \
    --disable-shared \
    --enable-static \
    --disable-doc \
    --disable-debug \
    --disable-ffplay \
    --disable-ffprobe \
    --disable-avdevice \
    --disable-symver \
    --extra-cflags="-Os -fpic" \
    --extra-ldflags=""

  make -j$(nproc)
  make install

  echo "→ Installed to $PREFIX/bin/ffmpeg"
done

echo "All done! Your binaries live in:"
ls -1 $OUTPUT_DIR/*/bin/ffmpeg