#include <mpi/mpi.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>

int myid, numprocs, ROOT = 0;
int *sendcounts, *displs;
double *AB, *A, *X;

void Iter_Jacoby(double *X_old, int size, int MATR_SIZE, int first) {
  double Sum;
  for (int i = 0; i < size; ++i) {
    Sum = 0;
    for (int j = 0; j < i + first; ++j)
      Sum += A[i * (MATR_SIZE + 1) + j] * X_old[j];
    for (int j = i + 1 + first; j < MATR_SIZE; ++j) {
      Sum += A[i * (MATR_SIZE + 1) + j] * X_old[j];
    }
    X[i + first] = (A[i * (MATR_SIZE + 1) + MATR_SIZE] - Sum) /
                   A[i * (MATR_SIZE + 1) + i + first];
  }
}

void SolveSLAE(int MATR_SIZE, int size, double Error) {
  double *X_old;
  int Iter = 0, Result, first;
  double dNorm = 0, dVal;
  MPI_Scan(&size, &first, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  std::cout << first << '\n';
  first -= size;
  MPI_Allgather(&size, 1, MPI_INT, sendcounts, 1, MPI_INT, MPI_COMM_WORLD);
  displs[0] = 0;
  for (int i = 1; i < numprocs; ++i)
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  X_old = new double[MATR_SIZE];
  do {
    ++Iter;
    for (size_t i = 0; i < MATR_SIZE; i++) {
      X_old[i] = X[i];
    }
    Iter_Jacoby(X_old, size, MATR_SIZE, first);
    MPI_Allgatherv(&X[first], size, MPI_DOUBLE, X, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    if (myid == ROOT) {
      for (int i = 0; i < MATR_SIZE; ++i) {
        dVal = fabs(X[i] - X_old[i]);
        if (dNorm < dVal) dNorm = dVal;
      }
      Result = Error < dNorm;
      dNorm = 0;
    }
    MPI_Bcast(&Result, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  } while (Result);
  delete[] X_old;
}


int main(int argc, char *argv[]) {
  int size, MATR_SIZE, SIZE;
  double Error;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid == ROOT) {
    std::cout << "Solve using Jacoby method" << std::endl;
    std::ifstream fin("input.txt");
    fin >> MATR_SIZE;
    AB = new double[MATR_SIZE * (MATR_SIZE + 1)];
    for (size_t i = 0; i < MATR_SIZE; i++) {
      for (size_t j = 0; j < MATR_SIZE + 1; j++) {
        fin >> AB[i * (MATR_SIZE + 1) + j];
      }
    }
    Error = 0.000001;
    fin.close();
  }
  MPI_Bcast(&MATR_SIZE, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&Error, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  X = new double[MATR_SIZE];
  if (myid == ROOT) {
    for (size_t i = 0; i < MATR_SIZE; i++) {
      X[i] = 0.0;
    }
  }
  MPI_Bcast(X, MATR_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  size = (MATR_SIZE / numprocs) + ((MATR_SIZE % numprocs) > myid ? 1 : 0);
  A = new double[(MATR_SIZE + 1) * size];
  displs = new int[numprocs];
  sendcounts = new int[numprocs];
  SIZE = (MATR_SIZE + 1) * size;
  MPI_Gather(&SIZE, 1, MPI_INT, sendcounts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  displs[0] = 0;
  for (int i = 1; i < numprocs; ++i) {
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  }
  MPI_Scatterv(AB, sendcounts, displs, MPI_DOUBLE, A, (MATR_SIZE + 1) * size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  auto t1 = std::chrono::high_resolution_clock::now();
  SolveSLAE(MATR_SIZE, size, Error);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  char processorName[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(processorName, &len);
  std::cout << processorName << "(" << myid << ")" << ": " << duration << "ms" << std::endl;
  if (myid == ROOT) {
    std::cout << "(";
    for (size_t i = 0; i < MATR_SIZE; i++) {
      std::cout << X[i] << ", ";
    }
    std::cout << ")";
  }
  std::cout << "\n";
  MPI_Finalize();
  delete[] sendcounts;
  delete[] displs;
  delete[] AB;
  delete[] A;
  delete[] X;
  return 0;
}