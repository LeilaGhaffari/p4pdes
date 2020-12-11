#include <petsc.h>

int main(int argc, char **argv) {
    
    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscInt       i;
    PetscReal      localval, globalsum;
    PetscReal      x = 1; // default value
  
    ierr = PetscInitialize(&argc, &argv, NULL,
                           "Compute e in parallel with PETSc.\n\n"); 
    if (ierr) return ierr;
  
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, 
                             "Maclaurin Series for exp(x)",
                             NULL); CHKERRQ(ierr); 
      ierr = PetscOptionsReal("-x", "input to exp(x) function", 
                              NULL, x, &x, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // compute  x^n / n!  where n = (rank of process) + 1
    localval = 1.;
    for (i = 1; i < rank+1; i++) localval *= x/i;
  
    // sum the contributions over all processes
    ierr = MPI_Allreduce(&localval, &globalsum, 1, MPIU_REAL, MPIU_SUM,
                         PETSC_COMM_WORLD); CHKERRQ(ierr);
  
    // output estimate of e and report on work from each process
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "exp(x) is about %17.15f\n", 
                       globalsum); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,
                       "rank %d did %d flops\n", 
                       rank, (rank > 0) ? rank-1 : 0); CHKERRQ(ierr);

    return PetscFinalize();
}