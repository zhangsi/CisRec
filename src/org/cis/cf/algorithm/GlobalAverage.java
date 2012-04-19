package org.cis.cf.algorithm;

import org.cis.data.Ratings;

public class GlobalAverage implements RatingPredictor{

	private Ratings ratings;
	
	private double globalBias;
	
	public GlobalAverage(Ratings ratings){
		this.ratings = ratings;
	}
	
	@Override
	public double predict(int user_id, int item_id, boolean bound) {
		return this.globalBias;
	}

	@Override
	public void trainModel() {
		this.globalBias = ratings.averageRating();
	}
	
}
