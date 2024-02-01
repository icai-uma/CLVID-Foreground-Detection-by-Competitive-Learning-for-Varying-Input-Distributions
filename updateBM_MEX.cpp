/* 
Foreground Detection by Competitive Learning for Varying Input Distributions
International Journal of Neural Systems 
DOI: 10.1142/S0129065717500563  
Authors: Ezequiel López-Rubio, Miguel A. Molina-Cabello, Rafael M. Luque-Baena, Enrique Domínguez
Date: July 2018 
*/

/*
Use the following command to compile this MEX file at the Matlab (2016b) prompt:
mex -IEigen updateBM_MEX.cpp BasicMath.cpp
*/

#include "mex.h"
#include "BasicMath.h"
#include <iostream>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

/*---------------------------------------
 * Functions which return if the value is NaN or INF
 * They are needed because the Microsoft compiler (.NET 2008) does not include them as primitives 
 * (does not include the C99 especification)
 */
#ifndef isnan
bool isnan(double x) {
    return x != x;
}
#endif

#ifndef isinf
bool isinf(double x) {
    return ((x - x) != 0);
}
#endif
/* ------------------------------------------------------ */
#define REINITIALISATION_MODE 0

/* This is used for debugging porposes*/
#define DEBUG_MODE 0
#define MY_PIXEL 73389

/* Definition of pi number if it is not previously defined */
#ifndef M_PI
#define M_PI 3.141615
#endif



FILE * fich;

void PixelInitialisation(long NdxPixel,int Dimension,double *ptrMu,double *ptrNoise,double *ptrCounter,double *ptrPattern,int numNeurons);

class AlmacenDatos {
    public:
        int Dimension;      
        
        long NumPixels;
        const int *DimPatterns;
        int NumImageRows,NumImageColumns;

        int NumCompUnif,NumCompGauss,NumComp,CurrentFrame,Z,NumNeurons;
        double *ptrEpsilon,*ptrNumComp,*ptrNumCompUnif,*ptrNumCompGauss,*ptrPi,*ptrNumNeurons;
        double *ptrCounter,*ptrNoise,*ptrMuFore,*ptrZ,*ptrDistanceThreshold,*ptrCoefWinner,*ptrCoefAll;
        double *ptrCurrentFrame,*pDataResp,*ptrOutput,*ptrDistances;
        double LearningRate,OneLessLearningRate;
        double LearningRateFore,OneLessLearningRateFore,PiFore,*ptrVectorProd,*ptrVectorDif;
        double DistanceThreshold,CoefWinner,CoefAll;

        double *ptrMu,*ptrC,*ptrLogDetC,*ptrDen,*ptrMin,*ptrMax;
        int DimResp[3];
        double *ptrData;
        
        AlmacenDatos(mxArray **plhs, const mxArray **prhs){
            mxArray *Mu,*C,*LogDetC,*Min,*Max,*Den,*Counter,*MuFore;
            
             #if (DEBUG_MODE == 1) 
                /*Log variables*/
                char *fileName;
                mwSize loglen; 
                mxArray *Log;

                /* Getting the name of the log file*/
                Log = mxGetField(prhs[0],0,"Log");
                loglen = mxGetNumberOfElements(Log) + 1;
                fileName = malloc(loglen*sizeof(char));

                if (mxGetString(Log, fileName, loglen) != 0)
                    mexErrMsgTxt("Could not convert string data.");

                fich = OpenLog(fileName);
                fprintf(fich,"Beginning of the initialisation process\n");
            #endif

            /* Get input data */
            DimPatterns=mxGetDimensions(prhs[1]);
            NumImageRows = DimPatterns[0];
            NumImageColumns = DimPatterns[1];
            Dimension = DimPatterns[2];
            ptrData = (double *) mxGetData(prhs[1]);
            NumPixels = NumImageRows * NumImageColumns;

            /* Duplicate the model structure */
            plhs[0]=mxDuplicateArray(prhs[0]);

            /* Get the work variables */
            ptrNumCompGauss=mxGetPr(mxGetField(plhs[0],0,"NumCompGauss"));
            ptrNumCompUnif=mxGetPr(mxGetField(plhs[0],0,"NumCompUnif"));
            ptrNumComp=mxGetPr(mxGetField(plhs[0],0,"NumComp"));
            ptrEpsilon=mxGetPr(mxGetField(plhs[0],0,"Epsilon"));
            ptrNumNeurons=mxGetPr(mxGetField(plhs[0],0,"NumNeurons"));
            ptrDistanceThreshold=mxGetPr(mxGetField(plhs[0],0,"DistanceThreshold"));
            ptrCoefWinner=mxGetPr(mxGetField(plhs[0],0,"CoefWinner"));
            ptrCoefAll=mxGetPr(mxGetField(plhs[0],0,"CoefAll"));
            Mu=mxGetField(plhs[0],0,"Mu");
            Counter=mxGetField(plhs[0],0,"Counter");            

            NumCompGauss =(int)(*ptrNumCompGauss);
            NumCompUnif = (int)(*ptrNumCompUnif);
            NumComp=(int)(*ptrNumComp);
            NumNeurons= (int)(*ptrNumNeurons);
            DistanceThreshold= (double)(*ptrDistanceThreshold);
            CoefWinner= (double)(*ptrCoefWinner);
            CoefAll= (double)(*ptrCoefAll);

            /* Noise values for each dimension  */
            ptrNoise=mxGetPr(mxGetField(plhs[0],0,"Noise"));

            /* Work pointers */
            ptrMu = mxGetPr(Mu);
            ptrCounter = mxGetPr(Counter);            

            #if (REINITIALISATION_MODE == 1)  
                ptrZ=mxGetPr(mxGetField(plhs[0],0,"Z"));
                Z = (int)(*ptrZ);	
                #if (DEBUG_MODE == 1) 
                    fprintf(fich,"Z value for pixel reinitialisation: %d\n", Z);
                #endif
            #endif
                        
            /* Create a matrix for the return argument */
            DimResp[0]=DimPatterns[0];
            DimResp[1]=DimPatterns[1];
            DimResp[2] = NumComp;
            plhs[1] = mxCreateNumericArray(2,DimPatterns,mxDOUBLE_CLASS, mxREAL);
            plhs[2] = mxCreateNumericArray(3,DimResp,mxDOUBLE_CLASS, mxREAL);
			plhs[3] = mxCreateNumericArray(2,DimPatterns,mxDOUBLE_CLASS, mxREAL);

            /* Assign pointers to the outputs */
            ptrOutput = (double*) mxGetData(plhs[1]);
            pDataResp = (double*) mxGetData(plhs[2]);
			ptrDistances = (double*) mxGetData(plhs[3]);

            /* Update the current frame */
            ptrCurrentFrame=mxGetPr(mxGetField(plhs[0],0,"CurrentFrame")); 
            (*ptrCurrentFrame) = (*ptrCurrentFrame) + 1.0;
            CurrentFrame=(int)(*ptrCurrentFrame);

            #if (DEBUG_MODE == 1)  
                if (CurrentFrame == 1) {
                    fprintf(fich,"Beginning of the update process\n");
                    fprintf(fich,"Sequence noise: ");
                    RecordMatrixLog(fich,ptrNoise,1,Dimension); 
                }
                fprintf(fich,"Current frame nº: %d\n", CurrentFrame);	
            #endif

            /* Load the learning rate */
            LearningRate=(*ptrEpsilon);
            OneLessLearningRate=1.0-LearningRate;
            LearningRateFore=(*ptrEpsilon);
            OneLessLearningRateFore=1.0-LearningRateFore;
        }
};

class AplicadorParallel {
    public:
    AlmacenDatos *const mis_Datos;
    
    void run(int NumPixels) const {
        const AlmacenDatos *Datos = mis_Datos;
        double *aux_Pattern,*aux_ptrPi,*aux_ptrTempVector;
        double *aux_ptrCounter,*aux_ptrMuFore;
        double *aux_pDataResp,*aux_ptrOutput,*aux_ptrDistances;
        double aux_GaussianResponsibilities,aux_SumResponsibilities,aux_AntPi,aux_CoefOld,aux_CoefNew,aux_CoefOldFore,aux_CoefNewFore;
        double aux_PiFore,*aux_ptrVectorProd,*aux_ptrVectorDif,*aux_ptrResponsibilities;

        double aux_MyLogDensity;
        double aux_DistMahal;
        double *aux_ptrMu,*aux_ptrC,*aux_ptrLogDetC;
        double *aux_ptrData;
        double *aux_ptrSigma;
        Map<MatrixXd> *aux_Sigma;
        LLT<MatrixXd> *aux_MyLLT;
        Map<VectorXd> *aux_VectorDif;
        MatrixXd *aux_L;
        int aux_NdxDim,aux_NdxComp,aux_NdxPixel;
        
        double *SquaredDistances;
        double MinimumDistance;
        int NdxWinner;
        
        /* Allocate dynamic data */
        aux_Pattern=(double *) malloc(Datos->Dimension*sizeof(double));
		aux_ptrVectorProd=(double *) malloc(Datos->Dimension*Datos->Dimension*sizeof(double));
		aux_ptrVectorDif=(double *) malloc(Datos->Dimension*sizeof(double));
		aux_ptrTempVector=(double *) malloc(Datos->Dimension*sizeof(double));
		aux_ptrResponsibilities=(double *) malloc(Datos->NumComp*sizeof(double));
		aux_ptrSigma=(double *) malloc(Datos->Dimension*Datos->Dimension*sizeof(double));
		aux_Sigma=new Map<MatrixXd>(aux_ptrSigma,Datos->Dimension,Datos->Dimension);
		aux_VectorDif=new Map<VectorXd>(aux_ptrVectorDif,Datos->Dimension);
		aux_MyLLT=new LLT<MatrixXd>(Datos->Dimension);
		aux_L=new MatrixXd(Datos->Dimension,Datos->Dimension);
        SquaredDistances=(double *) malloc(Datos->NumNeurons*sizeof(double));
        
	    /* Pointers are incremented */
        aux_ptrMu=Datos->ptrMu;
        aux_ptrData=Datos->ptrData;
        aux_pDataResp=Datos->pDataResp;
        aux_ptrOutput=Datos->ptrOutput;
        aux_ptrCounter=Datos->ptrCounter;
		aux_ptrDistances=Datos->ptrDistances;        

        for (aux_NdxPixel=0; aux_NdxPixel<NumPixels; ++aux_NdxPixel)
		{
			for(aux_NdxDim=0;aux_NdxDim<Datos->Dimension;aux_NdxDim++)
			{
				aux_Pattern[aux_NdxDim]=aux_ptrData[aux_NdxDim*Datos->NumPixels];
			}
		
			/* The pixel is reinitialised if it belongs to the foreground too much time */
			#if (REINITIALISATION_MODE == 1)  
			if (*aux_ptrCounter > Datos->Z) PixelInitialisation(aux_NdxDim,Datos->Dimension,aux_ptrMu,Datos->ptrNoise,aux_ptrCounter,aux_Pattern,Datos->NumNeurons);
			#endif

			/* ------------------------------------------------------------------------------ */ 
			/* Start of the code to compute the responsibilities */
			/* ------------------------------------------------------------------------------ */

			/* Responsibilities of the components */
			aux_SumResponsibilities=0.0;
			for(aux_NdxComp=0;aux_NdxComp<Datos->NumNeurons;aux_NdxComp++)
			{				
				Difference(aux_Pattern,aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrVectorDif,Datos->Dimension);
                SquaredNorm(aux_ptrVectorDif,SquaredDistances+aux_NdxComp,Datos->Dimension);				
			}
            
            // Get the winner neuron. It has the minimum distance
            MinimumDistance=SquaredDistances[0];
            NdxWinner=0;
            for(aux_NdxComp=1;aux_NdxComp<Datos->NumNeurons;aux_NdxComp++)
			{
                if (SquaredDistances[aux_NdxComp]<MinimumDistance)
                {
                    NdxWinner=aux_NdxComp;
                    MinimumDistance=SquaredDistances[aux_NdxComp];
                }                
            }
			
			*(aux_ptrDistances) = MinimumDistance;
            
            // Classify the pixel into background or foreground
            if (MinimumDistance<Datos->DistanceThreshold)
            {
                *(aux_ptrOutput) = 1; // Pixel is background 
            }
            else
            {
                *(aux_ptrOutput) = 0; // Pixel is foreground
            }
                
			/* The counter is incremented if the pixel belongs to the foreground */
			if (*(aux_ptrOutput)==0) (*aux_ptrCounter)+=1;
			else (*aux_ptrCounter)=0;            
            

			/* ------------------------------------------------------------------------------ */
			/* End of the code to compute the responsibilities */
			/* ------------------------------------------------------------------------------ */
				

			/* ------------------------------------------------------------------------------ */ 
			/* Start of the code to update the model */
			/* ------------------------------------------------------------------------------ */
			
			/* If the pixel is background we update the neurons */
            /********/ /*if (*(aux_ptrOutput)==1){*/  /* if this is commented the model always learn, so L=true */
                /* Update of the parameters of the Gaussian distributions */ 
                for(aux_NdxComp=0;aux_NdxComp<Datos->NumNeurons;aux_NdxComp++) 
                {
                    if (aux_NdxComp==NdxWinner)
                    {
                        // Note that CoefAll is much smaller than CoefWinner
                        aux_CoefNew=Datos->CoefWinner+Datos->CoefAll;                    
                    }
                    else
                    {
                        aux_CoefNew=Datos->CoefAll;
                    }
                    aux_CoefOld = 1 - aux_CoefNew; /* CoefOld + CoefNew == 1.0 */
                    ScalarMatrixProduct(aux_CoefNew,aux_Pattern,aux_ptrTempVector,Datos->Dimension,1);
                    ScalarMatrixProduct(aux_CoefOld,aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrMu+aux_NdxComp*Datos->Dimension,Datos->Dimension,1);
                    MatrixSum(aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrTempVector,aux_ptrMu+aux_NdxComp*Datos->Dimension,Datos->Dimension,1);
                }
            /*}*/ /********/

			/* ------------------------------------------------------------------------------ */
			/* End of the code to update the model */
			/* ------------------------------------------------------------------------------ */

			
			/* Pointers are incremented */		
			aux_ptrMu+=Datos->NumNeurons*Datos->Dimension;			
			aux_ptrData++;
			aux_pDataResp++;
			aux_ptrOutput++;
			aux_ptrCounter++;
			aux_ptrDistances++;
		}			
             
         /* Release pointers */       
        free(aux_Pattern);
		free(aux_ptrVectorProd);
		free(aux_ptrVectorDif);
		free(aux_ptrTempVector);
		free(aux_ptrResponsibilities);
		free(aux_ptrSigma);
        free(SquaredDistances);
		delete aux_Sigma;
		delete aux_MyLLT;
		delete aux_VectorDif;
		delete aux_L;
    }
    
    AplicadorParallel(AlmacenDatos *const aux_Datos) : mis_Datos(aux_Datos)
    {
    }
        
    ~AplicadorParallel(){
        

    }
};

void mexFunction( int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs )     
{
	const int *DimPatterns;
	int NumImageRows,NumImageColumns;
	long NumPixels;
	/* Get input data */
    DimPatterns=mxGetDimensions(prhs[1]);
	NumImageRows = DimPatterns[0];
	NumImageColumns = DimPatterns[1];
	NumPixels = NumImageRows * NumImageColumns;
    
    
	/* For each one of the image pixels */
    AlmacenDatos *const Datos = new AlmacenDatos(plhs, prhs);
    Eigen::initParallel();
	
    AplicadorParallel *ap=new AplicadorParallel(Datos);
	ap->run(NumPixels);	
	delete ap;
    
	delete Datos;
}
 
/**********************************************************************************************
 * Function to reinitialise a pixel which has exceeded the Z value. Z is the maximun number of consecutive 
 * frames in which a pixel belongs to the foreground class.
 **********************************************************************************************/
void PixelInitialisation(long NdxPixel,int Dimension,double *aux_ptrMu,double *ptrNoise,double *ptrCounter,double *ptrPattern,int NumNeurons) 
{
	double tmpLogDetC;
	double *temp;
	int NdxDim;
    int aux_NdxComp;

	temp=(double *)malloc(Dimension*sizeof(double));

	/* The counter for this pixel is initialised*/
	*ptrCounter = 0;

    for(aux_NdxComp=0;aux_NdxComp<NumNeurons;aux_NdxComp++) 
	{
        memcpy(aux_ptrMu+aux_NdxComp*Dimension,ptrPattern,Dimension*sizeof(double));
    }		
}
