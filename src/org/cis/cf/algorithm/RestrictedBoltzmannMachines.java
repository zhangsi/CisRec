package org.cis.cf.algorithm;


import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;
import org.cis.util.*;

/**
 * This class implementing Restricted Boltzmann Machines for Collaborative Filtering
 * 
 * The origin paper:
 * 
 * Salakhutdinov, R., Mnih, A. Hinton, G, Restricted BoltzmanMachines for Collaborative Filtering, 
 * To appear inProceedings of the 24thInternational Conference onMachine Learning 2007.
 * http://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class RestrictedBoltzmannMachines implements RatingPredictor{

	/** training data set of ratings */
	Ratings ratings;
	/** test data set of ratings */
	Ratings testRatings;
	
	/** user number */
	int userNumber;
	/** item number */
	int itemNumber;
	
	/** number of latent factors in hidden layer */
	int featureNumber;
	/** 5 */
	int softmax;
	
	/** max iteration number */
	int maxIter;
	
	/** learning rate for parameters */
	double epsilonw;
	double epsilonvb;
	double epsilonhb;
	double weightCost;
	double momentum;
	double finalMomentum;
	
	/** training data set indexed by user */
	int[][] trainSet;
	
	/** test data set indexed by user */
	ArrayList<Integer>[] testSet;

	
	/** model parameters */
	double[][][] weights;
	double[][]   visbiases;
	double[]     hidbiases;
	
	double[][][] CDpos;
	double[][][] CDneg;
	double[][][] CDinc;
	
	double[] poshidact;
	double[] neghidact;
	char[]   poshidstates;
	char[]   neghidstates;
	double[] hidbiasinc;
	
	char[] curposhidstates;
	
	double[][] posvisact;
	double[][] negvisact;
	double[][] visbiasinc;
	double[][] negvisprobs;
	
	char[]   negvissoftmax;
	int[] moviecount;
	
	/**
	 * Construct RBM algorithm 
	 * 
	 * @param ratings
	 * @param testRatings
	 * @param featureNumber
	 * @param softmax
	 * @param maxIter
	 * @param epsilonw
	 * @param epsilonvb
	 * @param epsilonhb
	 * @param weightCost
	 * @param momentum
	 * @param finalMomentum
	 */
	public RestrictedBoltzmannMachines(
			Ratings ratings,
			Ratings testRatings,
			int featureNumber,
			int softmax,
			int maxIter,
			double epsilonw,
			double epsilonvb,
			double epsilonhb,
			double weightCost,
			double momentum,
			double finalMomentum
			){
		this.ratings = ratings;
		this.testRatings = testRatings;
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.featureNumber = featureNumber;
		this.softmax       = softmax;
		this.maxIter       = maxIter;
		
		this.epsilonw      = epsilonw;
		this.epsilonvb     = epsilonvb;
		this.epsilonhb     = epsilonhb;
		
		this.momentum      = momentum;
		this.finalMomentum = finalMomentum;
		
		
		weights   = new double[itemNumber][softmax][featureNumber];
		visbiases = new double[itemNumber][softmax];
		hidbiases = new double[featureNumber];
		
		CDpos = new double[itemNumber][softmax][featureNumber];
		CDneg = new double[itemNumber][softmax][featureNumber];
		CDinc = new double[itemNumber][softmax][featureNumber];
		
		poshidact    = new double[featureNumber];
		neghidact    = new double[featureNumber];
		poshidstates = new char[featureNumber];
		neghidstates = new char[featureNumber];
		hidbiasinc   = new double[featureNumber];
		
		curposhidstates = new char[featureNumber];
		
		posvisact  = new double[itemNumber][softmax];
		negvisact  = new double[itemNumber][softmax];
		visbiasinc = new double[itemNumber][softmax];
		negvisprobs = new double[itemNumber][softmax];
		
		negvissoftmax = new char[itemNumber];
		moviecount = new int[itemNumber];
		
		trainSet = new int[userNumber][];
		testSet  = new ArrayList[userNumber];
		for( int u = 0; u != userNumber; ++u)
			testSet[u] = new ArrayList<Integer>();
		convertData();
	}
	
	/**
	 * Convert training data and test data from Ratings to user indexed form
	 */
	private void convertData() {
		ArrayList<ArrayList<Integer>> userList = ratings.getIndicesByUser();
		for( int u = 0; u != userNumber; ++u){
			int index,size, item, rating;
			size = userList.get(u).size();
			trainSet[u] = new int[size];
			for(int i = 0; i != size; ++i){
				index  = userList.get(u).get(i);
				item   = ratings.getItem(index);
				rating = ratings.getRating(index);
				trainSet[u][i] = item * 10 + rating;
			}
		}
		ratings.clear();
		userList.clear();
		
		int testUserNumber = testRatings.totalUserNumber();
		ArrayList<ArrayList<Integer>> testUserList = testRatings.getIndicesByUser();
		for( int u = 0; u != userNumber; ++u){
			int index,size, item, rating;
			if( u < testUserNumber){
				size = testUserList.get(u).size();
				for (int i = 0; i != size; ++i) {
					index = testUserList.get(u).get(i);
					item = testRatings.getItem(index);
					rating = testRatings.getRating(index);
					testSet[u].add(item * 10 + rating);
				}
			}
		}
		testRatings.clear();
		testUserList.clear();
	}
	
	/**
	 * Init model parameters
	 */
	private void initModel(){
		int[][] moviecount = new int[itemNumber][softmax];
		ZeroSetter.zero(moviecount, itemNumber, softmax);
		
		for(int user = 0; user < userNumber; user++) {
			int num = trainSet[user].length;
			
			for(int j = 0; j < num; j++) {
				//movie
				int m = trainSet[user][j] / 10 - 1;
				//rating
				int r = trainSet[user][j] % 10 - 1;
				moviecount[m][r]++;
			}	
		}
		
		Random randn = new Random();
		/** Set initial weights */
		for(int i = 0; i < itemNumber; i++) {
			for(int j = 0; j < featureNumber; j++) {
				for(int k = 0; k < softmax; k++) {
					/** Normal Distribution */
					weights[i][k][j] = 0.02 * randn.nextDouble() - 0.01;	        	
				}
			}
		}

		/** Set initial biases */
		ZeroSetter.zero(hidbiases, featureNumber);
		
		for(int i = 0; i < itemNumber; i++) {
			int mtot = 0;
			for(int k = 0; k < softmax; k++) {
				mtot += moviecount[i][k];
			}
			for(int k = 0; k < softmax; k++) {
				if(mtot == 0){
					visbiases[i][k] = new Random().nextDouble() * 0.001;
				}else{
					visbiases[i][k] = Math.log(((double)moviecount[i][k])/((double)mtot));
				}
			}
		}
	}
	
	/** 
	 * Train RBM model
	 */
	public void trainModel() {
		
		initModel();
		
		int loopcount = 0;
		int tSteps = 1;
		Random randn = new Random();
		
		ZeroSetter.zero(CDinc, itemNumber, softmax, featureNumber);
		ZeroSetter.zero(visbiasinc, itemNumber, softmax);
		ZeroSetter.zero(hidbiasinc, featureNumber);
		
		while(loopcount < maxIter) {
			
			if ( loopcount >= 10 )
				tSteps = 3 + (loopcount-10) / 5;
			loopcount++;
			
			if ( loopcount > 5 )
	        	momentum = finalMomentum;
			
			Zero();
			System.out.println(loopcount);
			
			for(int user = 0; user < userNumber; user++) {
				
				int num = trainSet[user].length;
				double[] sumW = new double[featureNumber];
			    ZeroSetter.zero(sumW, featureNumber);
			    
			    ZeroSetter.zero(negvisprobs, itemNumber, softmax);
			    
			    for(int i = 0; i < num; i++) {
			    	int m = trainSet[user][i] / 10 - 1;
					int r = trainSet[user][i] % 10 - 1;
					moviecount[m]++;
					
					posvisact[m][r] += 1.0;
					
					for(int h = 0; h < featureNumber; h++) {
						sumW[h]  += weights[m][r][h];
					}
			    }
			    
			    for(int h = 0; h < featureNumber; h++) {
			    	double probs = 1.0 / (1.0 + Math.exp(-sumW[h] - hidbiases[h]));
			    	if(probs > randn.nextDouble() ) {
			    		poshidstates[h] = 1;
			            poshidact[h] += 1.0;
			    	} else {
			    		poshidstates[h] = 0;
			    	}
			    }
			    
			    
			    for(int h = 0; h < featureNumber; h++)
			    	curposhidstates[h] = poshidstates[h];
			    
			    
			    /** Make T steps of Contrastive Divergence */
			    int stepT = 0;
			    do {
			    	boolean finalTStep = (stepT+1 >= tSteps);
			    	
			    	for(int i = 0; i < num; i++) {
			    		int m = trainSet[user][i] / 10 - 1;
			    		
			    		for(int h = 0; h < featureNumber; h++) {
			    			if(curposhidstates[h] == 1) {
			    				for(int r = 0; r < softmax; r++)
			    					negvisprobs[m][r]  += weights[m][r][h];
			    			}
			    		}
			    		
			    		for(int r = 0; r < softmax; r++)
			    			negvisprobs[m][r]  = 1./(1 + Math.exp(-negvisprobs[m][r] - visbiases[m][r]));
			    		
			    		/** Normalize probabilities */
			    		double tsum  = 0;
			    		for(int r = 0; r < softmax; r++) {
			    			tsum += negvisprobs[m][r];
			    		}
		
			    		if ( tsum != 0 ) {
			    			for(int r = 0; r < softmax; r++) {
			    				negvisprobs[m][r]  /= tsum;
			    			}
			    		}
			    		
			    		double randval = randn.nextDouble();
			    		
			            if ((randval -= negvisprobs[m][0]) <= 0.0)
			            	negvissoftmax[m] = 0;
			            else if ((randval -= negvisprobs[m][1]) <= 0.0)
			            	negvissoftmax[m] = 1;
			            else if ((randval -= negvisprobs[m][2]) <= 0.0)
			            	negvissoftmax[m] = 2;
			            else if ((randval -= negvisprobs[m][3]) <= 0.0)
			            	negvissoftmax[m] = 3;
			            else /** The case when ((randval -= negvisprobs[m][4]) <= 0.0) */			        	   
			            	negvissoftmax[m] = 4;
			    		
			    		if(finalTStep)
			    			negvisact[m][negvissoftmax[m]] += 1.0;
			    	}
			    	
			    	
			    	ZeroSetter.zero(sumW, featureNumber);
			    	for(int i = 0; i < num; i++) {
				    	int m = trainSet[user][i] / 10 - 1;
						
						for(int h = 0; h < featureNumber; h++) {
							sumW[h]  += weights[m][negvissoftmax[m]][h];
						}
				    }
				    
				    for(int h = 0; h < featureNumber; h++) {
				    	double probs = 1.0/(1.0 + Math.exp(-sumW[h] - hidbiases[h]));
				    	
				    	if(probs > randn.nextDouble() ) {
				    		neghidstates[h] = 1;
				    		if(finalTStep)
				    			neghidact[h] += 1.0;
				    	} else {
				    		neghidstates[h] = 0;
				    	}
				    }
				    
				    if(!finalTStep) {
				    	for(int h = 0; h < featureNumber; h++)
					    	curposhidstates[h] = neghidstates[h];
				    	ZeroSetter.zero(negvisprobs, itemNumber, softmax);
				    }	    	
			    	
			    } while ( ++stepT < tSteps );
			    
			    for(int i = 0; i < num; i++) {
			    	int m = trainSet[user][i] / 10 - 1;
					int r = trainSet[user][i] % 10 - 1;
					
					for(int h = 0; h < featureNumber; h++) {
						if ( poshidstates[h] == 1 ) {
			    			CDpos[m][r][h] += 1.0;
			    		}
			    		CDneg[m][negvissoftmax[m]][h] += (double)neghidstates[h];
					}
				}
			    
			    /** Update weights and biases */
			    update(user, num);
			}
			setArgument(loopcount);
			rmse();
		}
		
	}
	
	/**
	 * Update parameters
	 * 
	 * @param user
	 * @param num
	 */
	private void update(int user, int num) {
		
		/** Update weights and biases */
		 int bSize = 100;
		 if(((user + 1) % bSize)==0 || (user + 1) == userNumber) {
			 int numcases = user % bSize;
			 numcases++;
			 
			 /** Update weights */
			 for(int m = 0; m < itemNumber; m++) {
				 
				 if(moviecount[m] == 0)
					 continue;
				 
				 /** For all hidden units */
				 for(int h = 0; h < featureNumber; h++) {
					 
					 for(int r = 0; r < softmax; r++) {
						 double CDp = CDpos[m][r][h];
						 double CDn = CDneg[m][r][h];
						 if ( CDp != 0.0 || CDn != 0.0 ) {
							 CDp /= ((double)moviecount[m]);
							 CDn /= ((double)moviecount[m]);
	
		    					/** Update weights and biases W = W + alpha*ContrastiveDivergence (biases are just weights to neurons that stay always 1.0) */
		    					CDinc[m][r][h] = momentum * CDinc[m][r][h] + epsilonw * ((CDp - CDn) - weightCost * weights[m][r][h]);
		    					weights[m][r][h] += CDinc[m][r][h];
						 }
					 } 
				 }
				 
				 /** Update visible softmax biases */
				 for(int r = 0; r < softmax; r++) {
					 if(posvisact[m][r] != 0.0 || negvisact[m][r] != 0.0) {
						 posvisact[m][r] /= ((double)moviecount[m]);
						 negvisact[m][r] /= ((double)moviecount[m]);
						 visbiasinc[m][r] = momentum * visbiasinc[m][r] + epsilonvb * ((posvisact[m][r] - negvisact[m][r]));
						 visbiases[m][r]  += visbiasinc[m][r];
					 }
				 }
			 }	 
				 
			 /** Update hidden biases */
			 for(int h = 0; h < featureNumber; h++) {
				 if ( poshidact[h]  != 0.0 || neghidact[h]  != 0.0 ) {
					 poshidact[h]  /= ((double)(numcases));
					 neghidact[h]  /= ((double)(numcases));
					 hidbiasinc[h] = momentum * hidbiasinc[h] + epsilonhb * ((poshidact[h] - neghidact[h]));
					 hidbiases[h]  += hidbiasinc[h];
				 }
			 }
			 
			 Zero();
			 
		 }
	}
	
	/**
	 * Set the argument of the RBM model
	 * 
	 * @param loopcount
	 */
	private void setArgument(int loopcount) {
		if(featureNumber == 200) {
			 if ( loopcount > 6 ) {
				 epsilonw  *= 0.90;
				 epsilonvb *= 0.90;
				 epsilonhb *= 0.90;
			 } else if ( loopcount > 5 ) {  // With 200 hidden variables, you need to slow things down a little more
				 epsilonw  *= 0.50;         // This could probably use some more optimization
				 epsilonvb *= 0.50;
				 epsilonhb *= 0.50;
			 } else if ( loopcount > 2 ) {
				 epsilonw  *= 0.70;
				 epsilonvb *= 0.70;
				 epsilonhb *= 0.70;
			 }
		} else {
			 if ( loopcount > 8 ) {
				 epsilonw  *= 0.92;
				 epsilonvb *= 0.92;
				 epsilonhb *= 0.92;
			 } else if ( loopcount > 6 ) { 
				 epsilonw  *= 0.90;        
	           	 epsilonvb *= 0.90;
	           	 epsilonhb *= 0.90;
			 } else if ( loopcount > 2 ) {
				 epsilonw  *= 0.78;
				 epsilonvb *= 0.78;
				 epsilonhb *= 0.78;
			 }
		}
	}
	
	/**
	 * Set the model parameters to zero
	 */
	private void Zero() {
		ZeroSetter.zero(CDpos, itemNumber, softmax, featureNumber);
		ZeroSetter.zero(CDneg, itemNumber, softmax, featureNumber);
		ZeroSetter.zero(poshidact, featureNumber);
		ZeroSetter.zero(neghidact, featureNumber);
		ZeroSetter.zero(posvisact, itemNumber, softmax);
		ZeroSetter.zero(negvisact, itemNumber, softmax);
		ZeroSetter.zero(moviecount, itemNumber);
	}
	
	/**
	 * Calculate RMSE for training data and test data 
	 */
	private void rmse() {
		double nrmse = 0, prmse = 0;
		int tc = 0,pc = 0; 
		
		double[][] negvisprobs = new double[itemNumber][softmax];
		double[]   poshidprobs = new double[featureNumber];
		
		for(int user = 0; user < userNumber; user++) {
			int trainNumber = trainSet[user].length;
			int testNumber  = testSet[user].size();
			
			tc += trainNumber;
			pc += testNumber;
			
			double[] sumW = new double[featureNumber];
			ZeroSetter.zero(sumW, featureNumber);
			ZeroSetter.zero(negvisprobs, itemNumber, softmax);
			
			for(int i = 0; i < trainNumber; i++) {
				int item = trainSet[user][i] / 10 - 1;
				int rate = trainSet[user][i] % 10 - 1;
				
				for(int h = 0; h < featureNumber; h++) {
					sumW[h] += weights[item][rate][h];
				}
			}
			
			
			for(int h = 0; h < featureNumber; h++) {
				poshidprobs[h] = 1.0 / (1.0 + Math.exp(0 - sumW[h] - hidbiases[h]));
			}
			
			for(int i = 0; i < trainNumber + testNumber; i++) {
				int item;
				if(i < trainNumber)
					item = trainSet[user][i] / 10 - 1;
				else
					item = testSet[user].get(i - trainNumber) / 10 - 1;
				for(int h = 0; h < featureNumber; h++) {
					for(int r = 0; r < softmax; r++){
						negvisprobs[item][r] += poshidprobs[h] * weights[item][r][h];
					}
				}
			
				
				for(int r = 0; r < softmax; r++){
					negvisprobs[item][r] = 1.0 / (1.0 + Math.exp(0 - negvisprobs[item][r] - visbiases[item][r]));
				}
				
				double tsum = 0;
				for(int r = 0; r < softmax; r++) {
					tsum += negvisprobs[item][r];
				}
			
				if(tsum != 0) {
					for(int r = 0; r < softmax; r++) {
						negvisprobs[item][r] /= tsum;
					}
				}
			}
			
			for(int i = 0; i < trainNumber; i++) {
				int item = trainSet[user][i] / 10 - 1;
				int rate = trainSet[user][i] % 10 - 1;
				
				double predict = 0;
				for(int r = 0; r < softmax; r++) {
					predict += r * negvisprobs[item][r];
				}
				double errors = rate - predict;
				nrmse += errors * errors;
				
			}
			
			for(int i = 0; i < testNumber; i++) {
				int item = testSet[user].get(i) / 10 - 1;
				int rate = testSet[user].get(i) % 10 - 1;
				
				double predict = 0;
				for(int r = 0; r < softmax; r++) {
					predict += r * negvisprobs[item][r];
				}

				double errors = rate - predict;
				prmse += errors * errors;
			}
		}
		
		System.out.println("Train Rmse: " + Math.sqrt(nrmse / tc) + "		Test Rmse: " + Math.sqrt(prmse / pc));
		
	}


	/**
	 * Predict the rating value with given user_id and item_id
	 */
	public double predict(int user_id, int item_id, boolean bound) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	
}
