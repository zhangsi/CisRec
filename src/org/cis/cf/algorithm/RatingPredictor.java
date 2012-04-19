package org.cis.cf.algorithm;

public interface RatingPredictor {
	
	public double predict(int user_id, int item_id, boolean bound);
}
