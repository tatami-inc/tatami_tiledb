include(FetchContent)

FetchContent_Declare(
  tatami_chunked
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_chunked
  GIT_TAG master # ^2.1.0
)

FetchContent_Declare(
  sanisizer 
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^0.1.1
)

FetchContent_MakeAvailable(tatami_chunked)
FetchContent_MakeAvailable(sanisizer)
