function(sm120_nvfp4_discover_torch)
  if(TORCH_ROOT AND NOT TORCH_CXX11_ABI STREQUAL "")
    return()
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  if(NOT TORCH_ROOT)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
      RESULT_VARIABLE _torch_root_status
      OUTPUT_VARIABLE _torch_root
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _torch_root_status EQUAL 0)
      message(FATAL_ERROR "Could not discover TORCH_ROOT from Python3_EXECUTABLE")
    endif()
    set(TORCH_ROOT "${_torch_root}" CACHE PATH "PyTorch package root" FORCE)
  endif()

  if(TORCH_CXX11_ABI STREQUAL "")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
      RESULT_VARIABLE _torch_abi_status
      OUTPUT_VARIABLE _torch_abi
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _torch_abi_status EQUAL 0)
      message(FATAL_ERROR "Could not discover PyTorch's CXX11 ABI setting")
    endif()
    set(TORCH_CXX11_ABI "${_torch_abi}" CACHE STRING
        "PyTorch CXX11 ABI value" FORCE)
  endif()
endfunction()
