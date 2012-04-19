package org.cis.cf.algorithm;

import java.util.ArrayList;

import org.cis.data.Ratings;

public class UserAverage implements RatingPredictor{

	private Ratings ratings;
	private int userNumber;
	private int trainNumber;
	
	private ArrayList<Integer> userRatingSum;
	private ArrayList<Integer> userRatingCount;
	
	private double globalBias;
	
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
	
	@Override
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

	@Override
	public double predict(int user_id, int item_id, boolean bound) {
		
		if(user_id >= userNumber)
			return globalBias;
		
		if(userRatingCount.get(user_id) == 0)
			return globalBias;
		
		return (double) userRatingSum.get(user_id) / userRatingCount.get(user_id);
	}

}
