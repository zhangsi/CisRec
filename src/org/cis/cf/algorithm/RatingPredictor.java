package org.cis.cf.algorithm;

/**
 * This interface defining functions of a rating predictor
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public interface RatingPredictor {
	
	/**
	 * Train the model of a rating predictor
	 */
	public void trainModel();
	
	/**
	 * Predict the rating value of with given user_id and item id
	 * @param user_id
	 * @param item_id
	 * @param bound whether of bound the predicted value into [minRating, maxRating]
	 * @return
	 */
	public double predict(int user_id, int item_id, boolean bound);
	
}
