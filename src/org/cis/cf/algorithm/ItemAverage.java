package org.cis.cf.algorithm;

import java.util.ArrayList;

import org.cis.data.Ratings;

public class ItemAverage implements RatingPredictor{

	private Ratings ratings;
	private int itemNumber;
	private int trainNumber;
	
	private ArrayList<Integer> itemRatingSum;
	private ArrayList<Integer> itemRatingCount;
	
	private double globalBias;
	
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

	@Override
	public double predict(int user_id, int item_id, boolean bound) {
		
		if(item_id >= itemNumber)
			return globalBias;
		
		if(itemRatingCount.get(item_id) == 0)
			return globalBias;
		
		return (double) itemRatingSum.get(item_id) / itemRatingCount.get(item_id);
	}


}
