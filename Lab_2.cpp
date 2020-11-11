#include <fstream>
#include <iostream>
#include "mpi.h"
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>

std::vector<double> ReadMatrix() {
  int n = 500;

  std::vector<double> matrix(n * n);

  srand(time(0));
  for (int i = 0; i < n * n; ++i) {
    matrix[i] = rand() % n + 1;
  }

  return matrix;
}

void Transpose(std::vector<double> *matrix) {
  int n = std::sqrt(matrix->size());

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      std::swap(matrix->at(i * n + j), matrix->at(j * n + i));
    }
  }
}

void PrintMatrix(std::ofstream &fout, const std::vector<double> &matrix) {
  int n = std::sqrt(matrix.size());

  fout << n << '\n';

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      fout << matrix[i * n + j] << " ";
    }

    fout << '\n';
  }
}

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  std::cout << std::fixed;
  std::cout << std::setprecision(10);

  int rank;
  int size;
  int n;

  std::vector<double> A;

  MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    A = ReadMatrix();
    n = std::sqrt(A.size());

    Transpose(&A);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    A = std::vector<double>(n * n);
  }

  MPI_Bcast(&A[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double start = MPI_Wtime();

  for (int row = 0; row < n; ++row) {
    if (row % size == rank) {
      for (int next_row = row + 1; next_row < n; ++next_row) {
        if (A[row * n + next_row] != 0) {
          A[row * n + next_row] /= A[row * n + row];
        }
      }

      MPI_Send(&A[row * n], n, MPI_DOUBLE, 0, row + 69, MPI_COMM_WORLD);
    }

    MPI_Bcast(&A[row * n + row + 1], n - (row + 1), MPI_DOUBLE, row % size, MPI_COMM_WORLD);

    for (int column = row + 1; column < n; ++column) {
      if (column % size == rank) {
        for (int next_row = row + 1; next_row < n; ++next_row) {
          A[column * n + next_row] -= A[column * n + row] * A[row * n + next_row];
        }
      }
    }
  }

  double finish = MPI_Wtime();

  if (rank == 0) {
    for (int row = 0; row < n; ++row) {
      MPI_Recv(&A[row * n], n, MPI_DOUBLE, row % size, row + 69, MPI_COMM_WORLD, &status);
    }

    Transpose(&A);

    std::ofstream fout("output.txt");

    fout << std::fixed;
    fout << std::setprecision(6);

    PrintMatrix(fout, A);

    std::cout << "Processing time: " << finish - start << '\n';
  }

  MPI_Finalize();

  return 0;
}
