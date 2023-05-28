cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG ce677ad7fc170c8d81b0641a913d77f7418bc06f
)

FetchContent_MakeAvailable(tatami)
