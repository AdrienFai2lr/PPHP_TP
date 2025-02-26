#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define PAS 200
#define NBELEM 2000
#define ITA 3
#define ITB 2
#define ITC 2
#define ITD 1
#define ITE 1

int M1[NBELEM][NBELEM], M2[NBELEM][NBELEM], M3[NBELEM][NBELEM], 
    M4[NBELEM][NBELEM], M5[NBELEM][NBELEM], M6[NBELEM][NBELEM];
int A[NBELEM][NBELEM], B[NBELEM][NBELEM], C[NBELEM][NBELEM], 
    D[NBELEM][NBELEM], E[NBELEM][NBELEM];

// Initialisation des matrices avec des 1
void init2D(int tab[NBELEM][NBELEM]) {
    int i,j;
    for(i=0; i<NBELEM; i++) {
        for(j=0; j<NBELEM; j++) {
            tab[i][j] = 1;
        }
    }
}

// Mise à zéro des matrices
void raz2D(int tab[NBELEM][NBELEM]) {
    int i,j;
    for(i=0; i<NBELEM; i++) {
        for(j=0; j<NBELEM; j++) {
            tab[i][j] = 0;
        }
    }
}

int main(void) {
    int ncores;
    int i,j,k;
    long long res_seq = 0, res_par = 0, res_parV2 = 0, res_parV3 = 0;
    double start, end, tempsSeq, tempsParallel, tempsParallelV2, tempsParallelV3;
    
    ncores = omp_get_num_procs();
    printf("%d threads disponibles\n", ncores);
    
    // Initialisation des matrices
    init2D(M1); init2D(M2); init2D(M3); 
    init2D(M4); init2D(M5); init2D(M6);
    
    printf("\n=== Version séquentielle ===\n");
    raz2D(A); raz2D(B); raz2D(C); raz2D(D); raz2D(E);
    
    start = omp_get_wtime();
    
    // Traitement A
    for(i=0; i<PAS*ITA; i++) {
        for(j=0; j<PAS*ITA; j++) {
            for(k=0; k<PAS*ITA; k++) {
                A[i][j] = A[i][j] + M1[i][k]*M2[k][j];
            }
        }
    }
    
    // Traitement B
    for(i=0; i<PAS*ITB; i++) {
        for(j=0; j<PAS*ITB; j++) {
            for(k=0; k<PAS*ITB; k++) {
                B[i][j] = B[i][j] + M3[i][k]*M4[k][j];
            }
        }
    }
    
    // Traitement C
    for(i=0; i<PAS*ITC; i++) {
        for(j=0; j<PAS*ITC; j++) {
            for(k=0; k<PAS*ITC; k++) {
                C[i][j] = C[i][j] + M5[i][k]*M6[k][j];
            }
        }
    }
    
    // Traitement D
    for(i=0; i<PAS*ITD; i++) {
        for(j=0; j<PAS*ITD; j++) {
            for(k=0; k<PAS*ITD; k++) {
                D[i][j] = D[i][j] + B[i][k]*C[k][j];
            }
        }
    }
    
    // Traitement E
    for(i=0; i<PAS*ITE; i++) {
        for(j=0; j<PAS*ITE; j++) {
            for(k=0; k<PAS*ITE; k++) {
                E[i][j] = E[i][j] + A[i][k]*D[k][j];
            }
        }
    }
    
    end = omp_get_wtime();
    tempsSeq = (end-start);
    
    // Calcul résultat séquentiel
    for(i=0; i<NBELEM; i++) {
        for(j=0; j<NBELEM; j++) {
            res_seq = res_seq + E[i][j];
        }
    }
    printf("Temps séquentiel : %g secondes\n", tempsSeq);
    printf("Résultat séquentiel : %llu\n", res_seq);
    




    printf("\n=== Version parallèle V1 ===\n");
    // Réinitialisation des matrices pour version parallèle
    raz2D(A); raz2D(B); raz2D(C); raz2D(D); raz2D(E);
    
    start = omp_get_wtime();
    omp_set_nested(1);
    // Premier bloc : Calcul parallèle de A, B et C
    #pragma omp parallel sections private(i,j,k)
    {
        #pragma omp section
        {
            // Calcul de A
            for(i=0; i<PAS*ITA; i++) {
                for(j=0; j<PAS*ITA; j++) {
                    for(k=0; k<PAS*ITA; k++) {
                        A[i][j] = A[i][j] + M1[i][k]*M2[k][j];
                    }
                }
            }
        }
            
        #pragma omp section
        {
            // Calcul de B
            for(i=0; i<PAS*ITB; i++) {
                for(j=0; j<PAS*ITB; j++) {
                    for(k=0; k<PAS*ITB; k++) {
                        B[i][j] = B[i][j] + M3[i][k]*M4[k][j];
                    }
                }
            }
        }
            
        #pragma omp section
        {
            // Calcul de C
            for(i=0; i<PAS*ITC; i++) {
                for(j=0; j<PAS*ITC; j++) {
                    for(k=0; k<PAS*ITC; k++) {
                        C[i][j] = C[i][j] + M5[i][k]*M6[k][j];
                    }
                }
            }
        }
    }
    
    // Deuxième bloc : Calcul séquentiel de D et E
    // Calcul de D
    for(i=0; i<PAS*ITD; i++) {
        for(j=0; j<PAS*ITD; j++) {
            for(k=0; k<PAS*ITD; k++) {
                D[i][j] = D[i][j] + B[i][k]*C[k][j];
            }
        }
    }
    
    // Calcul de E
    for(i=0; i<PAS*ITE; i++) {
        for(j=0; j<PAS*ITE; j++) {
            for(k=0; k<PAS*ITE; k++) {
                E[i][j] = E[i][j] + A[i][k]*D[k][j];
            }
        }
    }
    
    end = omp_get_wtime();
    tempsParallel = (end-start);
    
    // Calcul résultat parallèle
    for(i=0; i<NBELEM; i++) {
        for(j=0; j<NBELEM; j++) {
            res_par = res_par + E[i][j];
        }
    }
    
    // Affichage des résultats et comparaisons
    printf("Temps parallèle : %g secondes\n", tempsParallel);
    printf("Résultat parallèle : %llu\n", res_par);
    printf("\n=== Comparaisons ===\n");
    printf("Accélération : %g\n", tempsSeq/tempsParallel);
    printf("Efficacité : %g\n", (tempsSeq/tempsParallel)/3);
    printf("Vérification résultats : %s\n", (res_seq == res_par) ? "OK" : "ERREUR");
    



    // version parallèle v2
    // dans une première section parallèle B et C s'éxecutent
    // dans une seconde section parallèle A et D s'éxecutent 
    // dans la dernière section, le calcul de E est ensuite exécuté en séquentiel 

    printf("\n=== Version parallèle V2 ===\n");
    // Réinitialisation des matrices 
    raz2D(A); raz2D(B); raz2D(C); raz2D(D); raz2D(E);
    
    start = omp_get_wtime();
    
    // Premier bloc : Calcul parallèle de B et C
    #pragma omp parallel sections private(i,j,k)
    {           
        #pragma omp section
        {
            // Calcul de B
            for(i=0; i<PAS*ITB; i++) {
                for(j=0; j<PAS*ITB; j++) {
                    for(k=0; k<PAS*ITB; k++) {
                        B[i][j] = B[i][j] + M3[i][k]*M4[k][j];
                    }
                }
            }
        }
            
        #pragma omp section
        {
            // Calcul de C
            for(i=0; i<PAS*ITC; i++) {
                for(j=0; j<PAS*ITC; j++) {
                    for(k=0; k<PAS*ITC; k++) {
                        C[i][j] = C[i][j] + M5[i][k]*M6[k][j];
                    }
                }
            }
        }
    }

    // Deuxième bloc : exécution en parallèle de A et D   
    #pragma omp parallel sections private(i,j,k)
    {
        
        #pragma omp section
        {
            // Calcul de A
            for(i=0; i<PAS*ITA; i++) {
                for(j=0; j<PAS*ITA; j++) {
                    for(k=0; k<PAS*ITA; k++) {
                        A[i][j] = A[i][j] + M1[i][k]*M2[k][j];
                    }
                }
            }
        }
                        
        
        #pragma omp section 
        {
            // Calcul de D
            for(i=0; i<PAS*ITD; i++) {
                for(j=0; j<PAS*ITD; j++) {
                    for(k=0; k<PAS*ITD; k++) {
                    D[i][j] = D[i][j] + B[i][k]*C[k][j];
                }
            }
        }
    
    }
    
    }
    
    // Enfin on fait en séquentiel le calcul de E
    for(i=0; i<PAS*ITE; i++) {
        for(j=0; j<PAS*ITE; j++) {
            for(k=0; k<PAS*ITE; k++) {
                E[i][j] = E[i][j] + A[i][k]*D[k][j];
            }
        }
    }
    
    end = omp_get_wtime();
    tempsParallelV2 = (end-start);
    
    // Calcul résultat 
    for(i=0; i<NBELEM; i++) {
        for(j=0; j<NBELEM; j++) {
            res_parV2 = res_parV2 + E[i][j];
        }
    }
    
    printf ("\n === temps et résultat ===\n");
    // Affichage des résultats et comparaisons
    printf("Temps parallèle V2: %g secondes\n", tempsParallelV2);
    printf("Résultat parallèle V2 : %llu\n", res_parV2);
    printf("Vérification résultats : %s\n", (res_seq == res_parV2) ? "OK" : "ERREUR");

    printf("\n=== Comparaisons avec la version séquentielle ===\n");
    printf("Accélération V2: %g\n", tempsSeq/tempsParallelV2);
    printf("Efficacité V2 : %g\n", (tempsSeq/tempsParallelV2)/3);

    
   
   

    printf("\n=== Version parallèle V3 ===\n");
    // Réinitialisation des matrices pour version parallèle
    raz2D(A); raz2D(B); raz2D(C); raz2D(D); raz2D(E);
    
    start = omp_get_wtime();
    omp_set_nested(1);
    // Premier bloc : Calcul parallèle de A, B et C
    #pragma omp parallel sections private(i,j,k)
    {
        #pragma omp section
        {
            // Calcul de A
            for(i=0; i<PAS*ITA; i++) 
            {
                for(j=0; j<PAS*ITA; j++)
                {
                    for(k=0; k<PAS*ITA; k++) 
                    {
                        A[i][j] = A[i][j] + M1[i][k]*M2[k][j];
                    }
                }
            }
        }

        #pragma omp section 
        {
            #pragma omp parallel sections private(i,j,k)
            {
                #pragma omp section
                {
                    // Calcul de B
                    for(i=0; i<PAS*ITB; i++) 
                    {
                        for(j=0; j<PAS*ITB; j++)
                        {
                            for(k=0; k<PAS*ITB; k++) 
                            {
                                B[i][j] = B[i][j] + M3[i][k]*M4[k][j];
                            }
                        }
                    }
                }

                #pragma omp section
                {
                    // Calcul de C
                    for(i=0; i<PAS*ITC; i++) 
                    {
                        for(j=0; j<PAS*ITC; j++) 
                        {
                            for(k=0; k<PAS*ITC; k++) 
                            {
                                C[i][j] = C[i][j] + M5[i][k]*M6[k][j];
                            }
                        }
                    }
                }            
            }
            // Calcul de D
            for(i=0; i<PAS*ITD; i++) {
                for(j=0; j<PAS*ITD; j++) {
                    for(k=0; k<PAS*ITD; k++) {
                        D[i][j] = D[i][j] + B[i][k]*C[k][j];
                    }
                }
            }
        }
    }
   
    // Calcul séquentiel de E
    for(i=0; i<PAS*ITE; i++) {
        for(j=0; j<PAS*ITE; j++) {
            for(k=0; k<PAS*ITE; k++) {
                E[i][j] = E[i][j] + A[i][k]*D[k][j];
            }
        }
    }
    
    end = omp_get_wtime();
    tempsParallelV3 = (end-start);
    
    // Calcul résultat 
    for(i=0; i<NBELEM; i++) {
        for(j=0; j<NBELEM; j++) {
            res_parV3 = res_parV3 + E[i][j];
        }
    }
    
    
    // Affichage des résultats et comparaisons
    printf("Temps v3: %g secondes\n", tempsParallelV3);
    printf("Résultat v3 : %llu\n", res_parV3);
    printf("\n=== Comparaisons ===\n");
    printf("Accélération v3: %g\n", tempsSeq/tempsParallelV3);
    printf("Efficacité v3 : %g\n", (tempsSeq/tempsParallelV3)/3);
    printf("Vérification résultats : %s\n", (res_seq == res_parV3) ? "OK" : "ERREUR");
    

    return 0;

}