package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;


public class BiasedBaseline implements RatingPredictor {
	private Ratings ratings;
	
	private int userNumber;
	private int itemNumber;
	
	private double globalBias;
	
	private double[] userBias;
	private double[] itemBias;
	
	private double maxIterNumber;
	private int trainNumber;
	
	private double learnRate;
	
	private int maxRating;
	private int minRating;
	
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
	
	private void initModel() {
		Random rand = new Random();
		for( int u = 0; u <= userNumber; ++u){
			userBias[u] = rand.nextGaussian() * 0.01;
		}
		
		for( int i = 0; i <= itemNumber; ++i){
			itemBias[i] = rand.nextGaussian() * 0.01;
		}
	}
	
	@Override
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
	
	public void trainModel() {
		initModel();
		learnBias();
	}
	
	private void learnBias(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate(ratings.getRandomIndex());
		}
	}
	
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
