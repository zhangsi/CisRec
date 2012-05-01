package org.cis.cf.algorithm;

import java.util.ArrayList;

import org.cis.data.Ratings;

public class UserAverage implements RatingPredictor{

	/** training data set of ratings */
	private Ratings ratings;
	/** number of users */
	private int userNumber;
	/** number of training ratings */
	private int trainNumber;
	
	/** the sum of ratings for each user */
	private ArrayList<Integer> userRatingSum;
	/** the count of ratings for each user */
	private ArrayList<Integer> userRatingCount;
	/** global rating average */ 
	private double globalBias;
	
	/**
	 * Construct UserAverage algorithm
	 * 
	 * @param ratings training ratings
	 */
	public UserAverage(Ratings ratings) {
		this.ratings     = ratings;
		this.userNumber  = ratings.totalUserNumber();
		this.trainNumber = ratings.getCount();
		this.globalBias  = ratings.averageRating();
		
		userRatingSum   = new ArrayList<Integer>();
		userRatingCount = new ArrayList<Integer>();
		for(int u = 0; u <= userNumber; ++u){
			userRatingSum.add(0);
			userRatingCount.add(0);
		}	
	}
	
	/**
	 * Train the model of Item Average
	 */
	public void trainModel() {
		int index;
		int user_id, rating;
		for( index = 0; index != trainNumber; ++index){
			user_id = ratings.getUser(index);
			rating  = ratings.getRating(index);
			
			userRatingSum.set(user_id, userRatingSum.get(user_id) + rating);
			userRatingCount.set(user_id, userRatingCount.get(user_id) + 1);
		}
	}

	/**
	 * Predict the rating value with given user and item
	 */
	public double predict(int user_id, int item_id, boolean bound) {
		
		if(user_id >= userNumber)
			return globalBias;
		
		if(userRatingCount.get(user_id) == 0)
			return globalBias;
		
		return (double) userRatingSum.get(user_id) / userRatingCount.get(user_id);
	}

}
