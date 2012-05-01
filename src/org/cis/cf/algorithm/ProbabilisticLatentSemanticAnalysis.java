package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;


/**
 * This class implementing the Probabilistic Latent Semantic Analysis for Collaborative Filtering algorithm
 * 
 * The origin paper:
 * 
 * T. Hofmann, Latent Semantic Models for Collaborative Filtering,
 * ACM Transactions on Information Systems 22 (2004), 89C115.
 * http://comminfo.rutgers.edu/~muresan/IR/Docs/Articles/toisHofmann2004.pdf
 * 
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class ProbabilisticLatentSemanticAnalysis implements RatingPredictor {
	
	/** training data set of ratings */
	Ratings ratings;
	
	/** number of users */
	int userNumber;
	/** number of items */
	int itemNumber;
	
	/** number of hidden variables z */
	int hidVariables;
	
	/** the grade of rating, such as "5" in netflix */
	int rating;
	
	/** max iteration number */
	int maxIterNumber;

	/** the beta parameter */
	double beta;
	
	/** P(Z|U) */
	double[][] Puz; 
	
	/** Q(Z|U,Y,T)  U:user Y:item T:rate, Z:hidden variables */
	
	double[][][] Q;
	
	/** mean and  variables for gaussian */
	double[][][] ud2yz;  	


	/** the training data set indexed by user */
	int[][] userInfo;
	/** the training data set indexed by item */ 
	int[][] itemInfo;
	
	/**
	 * Construct PLSA algorithm
	 * 
	 * @param ratings
	 * @param hidVariables
	 * @param rating
	 * @param beta
	 * @param maxIterNumber
	 */
	public ProbabilisticLatentSemanticAnalysis(Ratings ratings, int hidVariables, int rating, double beta, int maxIterNumber){	
		this.ratings = ratings;
		this.hidVariables = hidVariables;
		this.rating = rating;
		this.maxIterNumber = maxIterNumber;
		this.beta = beta;
	
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		Puz = new double[userNumber + 1][hidVariables]; 
		
		Q = new double[userNumber + 1][][];
		
		ud2yz = new double[itemNumber + 1][hidVariables][2];  
		
		userInfo = new int[userNumber + 1][];
		itemInfo = new int[itemNumber + 1][];

		convertData();
	}
	
	/**
	 * Convert training data from ratings to user indexed data and item indexed data
	 */
	private void convertData() {
		ArrayList<ArrayList<Integer>> userList = ratings.getIndicesByUser();
		for( int u = 1; u <= userNumber; ++u){
			int index,size, item, rating;
			size = userList.get(u).size();
			Q[u] = new double[size][hidVariables];
			userInfo[u] = new int[size];
			for(int i = 0; i != size; ++i){
				index  = userList.get(u).get(i);
				item   = ratings.getItem(index);
				rating = ratings.getRating(index);
				userInfo[u][i] = item * 10 + rating;
			}
		}
		userList.clear();
		
		ArrayList<ArrayList<Integer>> itemList = ratings.getIndicesByItem();
		for( int i = 1; i <= itemNumber; ++i){
			int index, size, user, rating;
			size = itemList.get(i).size();
			itemInfo[i] = new int[size];
			for( int u = 0; u != size; ++u){
				index  = itemList.get(i).get(u);
				user   = ratings.getUser(index);
				rating = ratings.getRating(index);
				itemInfo[i][u] = user * 10 + rating;
			}
		}
		
		itemList.clear();
		ratings.clear();
	}
	
	/**
	 * Init the model parameters
	 */
	private void initModel(){
		Random random = new Random();
		
		// Init P(Z|U);
		for(int i = 1; i <= userNumber; i++) {
			double norm = 0;
			for(int j = 0; j < hidVariables; j++) {
				Puz[i][j] = random.nextDouble();
				norm += Puz[i][j];
			}
			
			for(int j = 0; j < hidVariables; j++) {
				Puz[i][j] /= norm;
			}
		}
		
		double max_d = (1 + rating) / 2;
		max_d = (1 - max_d) * (1 - max_d);

		for(int i = 1; i <= itemNumber; i++) {
			for(int j = 0; j < hidVariables; j++) {
				ud2yz[i][j][0] = random.nextDouble() * (rating - 1) + 1;
				ud2yz[i][j][1] = random.nextDouble() * max_d;
			}
		}
	}
	
	/**
	 * train the PLSA model
	 */
	public void trainModel() {
		initModel();
		learnParameters();
	}
	
	/**
	 * update parameter with given max iteration number
	 */
	private void learnParameters(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			eStep();
			mStep();
		}
	}
	
	/**
	 * E step of the PLSA algorithm
	 * @return
	 */
	private double eStep() {
		double ans = 0;
		for(int user = 1; user <= userNumber; user++) {
			int numRate = userInfo[user].length;
			
			for(int j = 0; j < numRate; j++) {
				int item = userInfo[user][j] / 10;
				int rate = userInfo[user][j] % 10;
				
				double norm = 0;
				for(int label = 0; label < hidVariables; label++) {
					double part = Pvyz(rate, item, label) * Puz[user][label];
					Q[user][j][label] = Math.pow(part, beta);
					norm += Q[user][j][label];
				}
				
				for(int label = 0; label < hidVariables; label++) {
					double temp = Q[user][j][label] / norm;
					Q[user][j][label] = temp;
					ans += temp;
				}
			}
		}
		return ans;
	}
	
	/**
	 * M step of the PLSA algorithm
	 */
	private void mStep() {
		
		//update P(Z|U)
		for(int user = 1; user <= userNumber; user++) {
			double norm = 0;
			for(int z = 0; z < hidVariables; z++) {
				Puz[user][z] = 0;
				for(int i = 0; i < userInfo[user].length; i++) {
					Puz[user][z] += Q[user][i][z];
				}
				norm += Puz[user][z];
			}
			
			for(int z = 0; z < hidVariables; z++) {
				Puz[user][z] /= norm;
			} 
		}
		
		
		
		for(int item = 1; item <= itemNumber; item++) {
			
			//update average
			for(int z = 0; z < hidVariables; z++) {
				double a = 0;
				double b = 0;
				
				for(int i = 0; i < itemInfo[item].length; i++) {
					int user = itemInfo[item][i] / 10;
					int rate = itemInfo[item][i] % 10;
					
					int index = 0;
					for(int j = 0; j < userInfo[user].length; j++) {
						if((userInfo[user][j] / 10) == i) {
							index = j;
							break;
						}
					}
					a += rate * Q[user][index][z];
					b += Q[user][index][z];
				}
				
				if(b != 0)
					ud2yz[item][z][0] = a / b;
				else {
					continue;
				}
			}
			
			//update variables
			for(int z = 0; z < hidVariables; z++) {
				double a = 0;
				double b = 0;
				
				for(int i = 0; i < itemInfo[item].length; i++) {
					int user = itemInfo[item][i] / 10;
					int rate = itemInfo[item][i] % 10;
					
					int index = 0;
					for(int j = 0; j < userInfo[user].length; j++) {
						if((userInfo[user][j] / 10) == i) {
							index = j;
							break;
						}
					}

					double dif = rate - ud2yz[item][z][0];
	
					a += dif * dif * Q[user][index][z];
					b += Q[user][index][z];
				}
				if(b != 0 && a != 0)
					ud2yz[item][z][1] = a / b;
				
			}
					
		}
		
	}
	
	/**
	 * compute the mean and variables of gaussian
	 * @param rate
	 * @param item
	 * @param label
	 * @return
	 */
	private double Pvyz(int rate, int item, int label) {
		
		double value = ud2yz[item][label][0];
		double variance = ud2yz[item][label][1];
		
		double temp = 0 - (rate - value) * (rate - value);
		
		double ans = 1 / Math.sqrt(Math.PI * 2 * variance ) * Math.exp(temp / 2 / variance);
		return ans;
	}

	/**
	 * Predict the rating value with given user and item
	 */
	public double predict(int user_id, int item_id, boolean bound) {
		double ans = 0;
		
		for(int z = 0; z < hidVariables; z++)
			ans += Puz[user_id][z] * ud2yz[item_id][z][0];
		return ans;
	}
	
}
