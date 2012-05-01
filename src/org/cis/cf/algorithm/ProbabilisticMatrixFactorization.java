package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * This class implementing the Probabilistic Matrix Factorization
 * 
 * the origin paper:
 * 
 * Salakhutdinov, R., & Mnih, A. (2008). Probabilistic matrix factorization. 
 * Advances in Neural Information Processing Systems 20. Cambridge, MA: MIT Press
 * http://www.cs.utoronto.ca/~amnih/papers/pmf.pdf	
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class ProbabilisticMatrixFactorization implements RatingPredictor{
	
	/** user factors */
	DenseDoubleMatrix2D userFeatures;
	/** item factors */
	DenseDoubleMatrix2D itemFeatures;
	
	/** training data set of ratings */
	Ratings ratings;
	
	/** max rating */
	int maxRating;
	/** min rating */
	int minRating;
	
	/** number of training ratings */
	int trainNumber;
	
	/** global average of all the ratings */ 
	double globalBias;
	
	/** learning rate of the model parameters */
	double learnRate;
	
	/** regularization of user factors */
	double userReg;
	/** regularization of item factors */
	double itemReg;
	
	/** number of latent factors */
	int featureNumber;
	/** max iteration number */
	int maxIterNumber;
	
	/** number of users */
	int userNumber;
	/** number of items */
	int itemNumber;
	
	/**
	 * @param learnRate the learnRate to set
	 */
	public void setLearnRate(double learnRate) {
		this.learnRate = learnRate;
	}

	/**
	 * @param userReg the userReg to set
	 */
	public void setUserReg(double userReg) {
		this.userReg = userReg;
	}

	/**
	 * @param itemReg the itemReg to set
	 */
	public void setItemReg(double itemReg) {
		this.itemReg = itemReg;
	}

	/**
	 * @param maxIterNumber the maxIterNumber to set
	 */
	public void setMaxIterNumber(int maxIterNumber) {
		this.maxIterNumber = maxIterNumber;
	}
	
	/**
	 * Construct PMF algorithm
	 * 
	 * @param ratings
	 * @param featureNumber
	 */
	public ProbabilisticMatrixFactorization(Ratings ratings, int featureNumber){
		this.ratings = ratings;
		this.globalBias = ratings.averageRating();
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.maxRating = ratings.getMaxRating();
		this.minRating = ratings.getMinRating();
		
		this.trainNumber = ratings.getCount();
		
		this.featureNumber = featureNumber;
		
		this.userFeatures = new DenseDoubleMatrix2D(userNumber + 1, featureNumber);
		this.itemFeatures = new DenseDoubleMatrix2D(itemNumber + 1, featureNumber);
	}
	
	/**
	 * Construct PMF algorithm
	 * 
	 * @param ratings
	 * @param featureNumber
	 * @param learnRate
	 * @param userReg
	 * @param itemReg
	 * @param maxIterNumber
	 */
	public ProbabilisticMatrixFactorization(Ratings ratings, int featureNumber, double learnRate,
			double userReg, double itemReg, int maxIterNumber) {
		this.ratings = ratings;
		this.globalBias = ratings.averageRating();
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.maxRating = ratings.getMaxRating();
		this.minRating = ratings.getMinRating();
		
		this.trainNumber = ratings.getCount();
		
		this.learnRate = learnRate;
		this.userReg   = userReg;
		this.itemReg   = itemReg;
		this.maxIterNumber = maxIterNumber;
		
		this.featureNumber = featureNumber;
		
		this.userFeatures = new DenseDoubleMatrix2D(userNumber + 1, featureNumber);
		this.itemFeatures = new DenseDoubleMatrix2D(itemNumber + 1, featureNumber);
	}
	
	/**
	 * Init the model parameters of PMF
	 */
	private void initModel(){
		Random rand = new Random();
		
		for( int u = 0; u != userNumber; ++u){
			for( int f = 0; f != featureNumber; ++f){
				userFeatures.setQuick(u, f, rand.nextGaussian() * 0.01);
			}
		}
		
		for( int i = 0; i != itemNumber; ++i){
			for( int f = 0; f != featureNumber; ++f){
				itemFeatures.setQuick(i, f, rand.nextGaussian() * 0.01);
			}
		}
	}
	
	/**
	 * Train the model of PMF
	 */
	public void trainModel(){
		
		initModel();
		learnFeatures();
	}
	
	/**
	 * Update the parameter with given max iteration number
	 */
	public void learnFeatures(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate(ratings.getRandomIndex());
		}
	}
	
	/**
	 * In a iteration loop, update the user factors and item factors
	 * @param list the randomly generated index list
	 */
	public void iterate(ArrayList<Integer> list){
		
		int user_id, item_id, rating;
		double err;
		for(int index = 0; index != trainNumber; ++index){
			
			user_id = ratings.getUser(index);
			item_id = ratings.getItem(index);
			rating  = ratings.getRating(index);
			
			err = rating - predict(user_id, item_id, false);
			
			// update factors
			for(int f = 0;  f != featureNumber; ++f){
				double u_f = userFeatures.getQuick(user_id, f);
				double i_f = itemFeatures.getQuick(item_id, f);
				
				double delta_u = err * i_f - userReg * u_f;
				userFeatures.setQuick(user_id, f, userFeatures.getQuick(user_id, f) + learnRate * delta_u);
				
				double delta_i = err * u_f - itemReg * i_f;
				itemFeatures.setQuick(item_id, f, itemFeatures.getQuick(item_id, f) + learnRate * delta_i);
			}
		}
	}
	
	/**
	 * Predict the rating value with given user_id and item_id
	 */
	public double predict(int user_id, int item_id, boolean bound){
		
		if(user_id >= userFeatures.rows())
			return this.globalBias;
		if(item_id >= itemFeatures.rows())
			return this.globalBias;
		
		Algebra algebra = new Algebra();
		double result = 0;
		result += globalBias;
		result += algebra.mult(userFeatures.viewRow(user_id), itemFeatures.viewRow(item_id));
		
		if(bound){
			if( result > maxRating)
				result = (double) maxRating;
			if( result < minRating)
				result = (double) minRating;
		}
		
		return result;
	}
}
