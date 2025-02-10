#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define PAS 200
#define NBELEM 2000
#define ITA 6
#define ITB 7
#define ITC 7
#define ITD 2
#define ITE 1

int M1[NBELEM][NBELEM], M2[NBELEM][NBELEM], M3[NBELEM][NBELEM], M4[NBELEM][NBELEM], M5[NBELEM][NBELEM], M6[NBELEM][NBELEM];

int A[NBELEM][NBELEM], B[NBELEM][NBELEM], C[NBELEM][NBELEM], D[NBELEM][NBELEM], E[NBELEM][NBELEM];

//initialisation des matrices initiales (valeur sans importance)
void init2D(int tab[NBELEM][NBELEM]){
    int i,j;
    for(i=0;i<NBELEM; i++){
	   for(j=0;j<NBELEM; j++){
			tab[i][j] = 1;
	   }
    }
}

//mise à 0 de la matrice résultat
void raz2D(int tab[NBELEM][NBELEM]){
    int i,j;
    for(i=0;i<NBELEM; i++){
	   for(j=0;j<NBELEM; j++){
			tab[i][j] = 0;
	   }
    }
}

int main (void){
	int ncores;
	int i,j,k;
	long long res;
	double start, end, tempsSeq;
	ncores = omp_get_num_procs();
	printf("%d coeurs disponibles\n",ncores);

	//version séquentielle

	init2D(M1);
	init2D(M2);
	init2D(M3);
	init2D(M4);
	init2D(M5);
	init2D(M6);

	raz2D(A);
	raz2D(B);
	raz2D(C);
	raz2D(D);
	raz2D(E);

	res=0;

	start=omp_get_wtime();

	//traitement A
	for(i=0;i<PAS*ITA;i++){
		for(j=0;j<PAS*ITA;j++){
			for(k=0;k<PAS*ITA;k++){
				A[i][j] = A[i][j]+M1[i][k]*M2[k][j];
			}
		}
	}

	//traitement B
	for(i=0;i<PAS*ITB;i++){
		for(j=0;j<PAS*ITB;j++){
			for(k=0;k<PAS*ITB;k++){
				B[i][j] = B[i][j]+M3[i][k]*M4[k][j];
			}
		}
	}

	//traitement C
	for(i=0;i<PAS*ITC;i++){
		for(j=0;j<PAS*ITC;j++){
			for(k=0;k<PAS*ITC;k++){
				C[i][j] = C[i][j]+M5[i][k]*M6[k][j];
			}
		}
	}

	//traitement D
	for(i=0;i<PAS*ITD;i++){
		for(j=0;j<PAS*ITD;j++){
			for(k=0;k<PAS*ITD;k++){
				D[i][j] = D[i][j]+B[i][k]*C[k][j];
			}
		}
	}

	//traitement E
	for(i=0;i<PAS*ITE;i++){
		for(j=0;j<PAS*ITE;j++){
			for(k=0;k<PAS*ITE;k++){
				E[i][j] = E[i][j]+A[i][k]*D[k][j];
			}
		}
	}

    end=omp_get_wtime();

	tempsSeq=(end-start);
	printf ("temps seq : %g \n",tempsSeq);

	//pour que les résultats soient utilisés
	for(i=0;i<NBELEM;i++){
		for(j=0;j<NBELEM;j++){
			res=res+E[i][j];
		}
	}
	printf ("resultat version seq: %I64d \n",res);


    return 0;
}
