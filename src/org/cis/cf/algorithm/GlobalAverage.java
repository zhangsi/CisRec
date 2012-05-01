package org.cis.cf.algorithm;

import org.cis.data.Ratings;

public class GlobalAverage implements RatingPredictor{

	/** training data set of ratings */
	Ratings ratings;
	/** global rating average */ 
	double globalBias;
	
	/**
	 * Construct the global average of the training data set
	 * @param ratings train ratings
	 */
	public GlobalAverage(Ratings ratings){
		this.ratings = ratings;
	}
	
	/**
	 * Predict the rating value with given user and item
	 */
	public double predict(int user_id, int item_id, boolean bound) {
		return this.globalBias;
	}

	/**
	 * Train the model of Global Average
	 */
	public void trainModel() {
		this.globalBias = ratings.averageRating();
	}
	
}
