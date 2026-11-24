#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <iterator>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
