package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;

/**
 * This class implementing the baseline biased model (equation 1 in SVD++ paper)
 * 
 * The origin paper:
 * 
 * Yehuda Koren., Factorization meets the neighborhood: A multifaceted collaborative filtering model. 
 * In Proceedings of the 14th ACM SIGKDD International Conference on
 * Knowledge Discovery and Data Mining (KDD'08) (2008), 426¨C434.
 * http://public.research.att.com/~volinsky/netflix/kdd08koren.pdf
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class BiasedBaseline implements RatingPredictor {
	/** training data set of ratings */
	Ratings ratings;
	
	/** number of users */
	int userNumber;
	/** number of items */
	int itemNumber;
	
	/** global rating average */ 
	double globalBias;
	
	/** the user bias parameter */
	double[] userBias;
	/** the item bias parameter */
	double[] itemBias;
	
	/** max iteration number */
	double maxIterNumber;
	/** the train ratings number */
	int trainNumber;
	
	/** learing rate for updating parameters */
	double learnRate;
	
	/** max rating */
	int maxRating;
	/** min rating */
	int minRating;
	
	/**
	 * Construct BiasedBaseline algorithm
	 * 
	 * @param ratings
	 * @param maxIterNumber
	 * @param learnRate
	 */
	public BiasedBaseline(Ratings ratings, int maxIterNumber, double learnRate){
		this.ratings = ratings;
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		this.globalBias = ratings.averageRating();
		this.maxIterNumber = maxIterNumber;
		this.trainNumber  = ratings.getCount();
		this.learnRate  = learnRate;
		
		this.maxRating = ratings.getMaxRating();
		this.minRating = ratings.getMinRating();
		
		userBias = new double[userNumber + 1];
		itemBias = new double[itemNumber + 1];
	}
	
	/**
	 * Init model parameters
	 */
	private void initModel() {
		Random rand = new Random();
		for( int u = 0; u <= userNumber; ++u){
			userBias[u] = rand.nextGaussian() * 0.01;
		}
		
		for( int i = 0; i <= itemNumber; ++i){
			itemBias[i] = rand.nextGaussian() * 0.01;
		}
	}
	
	/**
	 * Predict the rating value with given user and item
	 */
	public double predict(int user_id, int item_id, boolean bound){
		
		double result = userBias[user_id] + itemBias[item_id];
		result += globalBias;
		
		if(bound){
			if( result > maxRating)
				result = (double) maxRating;
			if( result < minRating)
				result = (double) minRating;
		}
		return result;
	}
	
	/**
	 * Train the BiasedBaseline model
	 */
	public void trainModel() {
		initModel();
		learnBias();
	}
	
	
	/**
	 * Learn the user and item bias with given max iteration number
	 */
	private void learnBias(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate(ratings.getRandomIndex());
		}
	}
	
	/**
	 * Update user bias and item bias in each iteration
	 */
	private void iterate(ArrayList<Integer> list){
		int user_id, item_id, rating;
		double prediction, gradient;
		for(int index = 0; index != trainNumber; ++index){
			
			user_id = ratings.getUser(index);
			item_id = ratings.getItem(index);
			rating  = ratings.getRating(index);
			
			prediction = globalBias + userBias[user_id] + itemBias[item_id];
			gradient = rating - prediction;
			
			userBias[user_id] += learnRate * learnRate * (gradient - learnRate  * userBias[user_id]);
			itemBias[item_id] += learnRate * learnRate * (gradient - learnRate  * itemBias[item_id]);
			
		}
	}
}
