package org.cis.cf.algorithm;

import java.util.ArrayList;

import org.cis.data.Ratings;

public class ItemAverage implements RatingPredictor{

	/** training data set of ratings */
	Ratings ratings;
	/** number of items */
	int itemNumber;
	/** number of training ratings */
	int trainNumber;
	
	/** the sum of ratings for each item */
	ArrayList<Integer> itemRatingSum;
	/** the count of ratings for each item */
	ArrayList<Integer> itemRatingCount;
	/** global rating average */ 
	double globalBias;
	
	/**
	 * Construct ItemAverage algorithm
	 * 
	 * @param ratings training ratings
	 */
	public ItemAverage(Ratings ratings) {
		this.ratings     = ratings;
		this.itemNumber  = ratings.totalItemNumber();
		this.trainNumber = ratings.getCount();
		this.globalBias  = ratings.averageRating();
		
		itemRatingSum   = new ArrayList<Integer>();
		itemRatingCount = new ArrayList<Integer>();
		for(int u = 0; u <= itemNumber; ++u){
			itemRatingSum.add(0);
			itemRatingCount.add(0);
		}	
	}
	
	/**
	 * Train the model of Item Average
	 */
	public void trainModel() {
		int index;
		int item_id, rating;
		for( index = 0; index != trainNumber; ++index){
			item_id = ratings.getItem(index);
			rating  = ratings.getRating(index);
			
			itemRatingSum.set(item_id, itemRatingSum.get(item_id) + rating);
			itemRatingCount.set(item_id, itemRatingCount.get(item_id) + 1);
		}
	}

	/**
	 * Predict the rating value with given user and item
	 */
	public double predict(int user_id, int item_id, boolean bound) {
		
		if(item_id >= itemNumber)
			return globalBias;
		
		if(itemRatingCount.get(item_id) == 0)
			return globalBias;
		
		return (double) itemRatingSum.get(item_id) / itemRatingCount.get(item_id);
	}


}
