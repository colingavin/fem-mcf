CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

INCLUDE_DIRECTORIES(/usr/local/opt/nlopt/include)
LINK_DIRECTORIES(/usr/local/opt/nlopt/lib)

FIND_PACKAGE(deal.II 8.4 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(RelaxationMCF)

set(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -Wno-unused-local-typedefs -Wno-mismatched-tags")

ADD_EXECUTABLE(CircleTestProblem 
  src/CSFSolver.cpp test/SolverDiagnostics.cpp test/CircleTestProblem.cpp)
DEAL_II_SETUP_TARGET(CircleTestProblem)

ADD_EXECUTABLE(DirBCTestProblem 
  src/CSFSolver.cpp test/SolverDiagnostics.cpp test/DirBCTestProblem.cpp)
DEAL_II_SETUP_TARGET(DirBCTestProblem)

ADD_EXECUTABLE(PaperclipTestProblem 
  src/CSFSolver.cpp test/SolverDiagnostics.cpp test/PaperclipTestProblem.cpp)
DEAL_II_SETUP_TARGET(PaperclipTestProblem)
TARGET_LINK_LIBRARIES(PaperclipTestProblem nlopt)

ADD_EXECUTABLE(SphereTestProblem 
  src/CSFSolver.cpp
  src/AxialMCFSolver.cpp
  test/SolverDiagnostics.cpp
  test/SphereTestProblem.cpp)
DEAL_II_SETUP_TARGET(SphereTestProblem)

ADD_EXECUTABLE(RoundToriTestProblem 
  src/CSFSolver.cpp
  src/AxialMCFSolver.cpp
  test/RoundToriTestProblem.cpp)
DEAL_II_SETUP_TARGET(RoundToriTestProblem)
