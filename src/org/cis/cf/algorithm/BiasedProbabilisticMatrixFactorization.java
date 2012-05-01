package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * The class combine the BiasedBaseline and PMF algorithm
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class BiasedProbabilisticMatrixFactorization implements RatingPredictor{
	
	/** user factors */
	DenseDoubleMatrix2D userFeatures;
	/** item factors */
	DenseDoubleMatrix2D itemFeatures;
	
	/** the user bias parameter */
	double[] userBias;
	/** the item bias parameter */
	double[] itemBias;
	
	/** training data set of ratings */
	Ratings ratings;
	
	/** max rating */
	int maxRating;
	/** min rating */
	int minRating;
	
	/** number of training ratings */
	int trainNumber;
	
	/** global bias of all the ratings */ 
	double globalBias;
	/** rating range from minRating to maxRating */
	double ratingRange;
	/** global average of all the ratings */ 
	double globalAvg;
	
	/** learning rate for factor parameters */
	double learnRate;
	/** regularization of user factors */
	double userReg;
	/** regularization of item factors */
	double itemReg;
	
	/** learning rate for bias parameters */
	double biasLearnRate;
	/** regularization of user bias */
	double biasUserReg;
	/** regularization of item bias */
	double biasItemReg;
	
	/** number of latent factors */
	int featureNumber;
	/** max iteration number */
	int maxIterNumber;
	
	/** number of users */
	int userNumber;
	/** number of items */
	int itemNumber;
	
	/**
	 * @param learnRate the learnRate to set
	 */
	public void setLearnRate(double learnRate) {
		this.learnRate = learnRate;
	}

	/**
	 * @param userReg the userReg to set
	 */
	public void setUserReg(double userReg) {
		this.userReg = userReg;
	}

	/**
	 * @param itemReg the itemReg to set
	 */
	public void setItemReg(double itemReg) {
		this.itemReg = itemReg;
	}

	/**
	 * @return the biasLearnRate
	 */
	public double getBiasLearnRate() {
		return biasLearnRate;
	}

	/**
	 * @param biasLearnRate the biasLearnRate to set
	 */
	public void setBiasLearnRate(double biasLearnRate) {
		this.biasLearnRate = biasLearnRate;
	}

	/**
	 * @param biasUserReg the biasUserReg to set
	 */
	public void setBiasUserReg(double biasUserReg) {
		this.biasUserReg = biasUserReg;
	}

	/**
	 * @param biasItemReg the biasItemReg to set
	 */
	public void setBiasItemReg(double biasItemReg) {
		this.biasItemReg = biasItemReg;
	}

	/**
	 * @param maxIterNumber the maxIterNumber to set
	 */
	public void setMaxIterNumber(int maxIterNumber) {
		this.maxIterNumber = maxIterNumber;
	}
	
	/**
	 * Construct Biased PMF algorithm
	 * 
	 * @param ratings
	 * @param featureNumber
	 */
	public BiasedProbabilisticMatrixFactorization(Ratings ratings, int featureNumber){
		this.ratings = ratings;

		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.maxRating = ratings.getMaxRating();
		this.minRating = ratings.getMinRating();
		
		this.ratingRange = this.maxRating - this.minRating;
		this.globalAvg = (ratings.averageRating() - this.minRating) / this.ratingRange;
		this.globalBias = Math.log( globalAvg / (1 - globalAvg));
		
		this.trainNumber = ratings.getCount();
		
		this.featureNumber = featureNumber;
		
		this.userFeatures = new DenseDoubleMatrix2D(userNumber + 1, featureNumber);
		this.itemFeatures = new DenseDoubleMatrix2D(itemNumber + 1, featureNumber);
		
		this.userBias     = new double[userNumber + 1];
		this.itemBias     = new double[itemNumber + 1];
	}
	
	/**
	 * Construct Biased PMF algorithm
	 * 
	 * @param ratings
	 * @param featureNumber
	 * @param learnRate
	 * @param userReg
	 * @param itemReg
	 * @param biasLearnRate
	 * @param biasUserReg
	 * @param biasItemReg
	 * @param maxIterNumber
	 */
	public BiasedProbabilisticMatrixFactorization(Ratings ratings, int featureNumber,
			double learnRate, double userReg, double itemReg, 
			double biasLearnRate, double biasUserReg, double biasItemReg,
			int maxIterNumber) {
		this.ratings = ratings;
		
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.maxRating = ratings.getMaxRating();
		this.minRating = ratings.getMinRating();
		
		this.ratingRange = this.maxRating - this.minRating;
		this.globalAvg = (ratings.averageRating() - this.minRating) / this.ratingRange;
		this.globalBias = Math.log( globalAvg / (1 - globalAvg));
		
		this.trainNumber = ratings.getCount();
		
		this.learnRate = learnRate;
		this.userReg   = userReg;
		this.itemReg   = itemReg;
		
		this.biasItemReg = biasItemReg;
		this.biasUserReg = biasUserReg;
		this.biasLearnRate = biasLearnRate;
		
		
		this.maxIterNumber = maxIterNumber;
		
		this.featureNumber = featureNumber;
		
		this.userFeatures = new DenseDoubleMatrix2D(userNumber + 1, featureNumber);
		this.itemFeatures = new DenseDoubleMatrix2D(itemNumber + 1, featureNumber);
		
		this.userBias     = new double[userNumber + 1];
		this.itemBias     = new double[itemNumber + 1];
	}
	
	/**
	 * Init the model parameters
	 */
	private void initModel(){
		Random rand = new Random();
		
		for( int u = 0; u != userNumber; ++u){
			for( int f = 0; f != featureNumber; ++f){
				userFeatures.setQuick(u, f, rand.nextGaussian() * 0.01);
			}
			userBias[u] = rand.nextGaussian() * 0.01;
		}
		
		for( int i = 0; i != itemNumber; ++i){
			for( int f = 0; f != featureNumber; ++f){
				itemFeatures.setQuick(i, f, rand.nextGaussian() * 0.01);
			}
			itemBias[i] = rand.nextGaussian() * 0.01;
		}
	}
	
	/**
	 * Train the Biased PMF model
	 */
	public void trainModel(){
		initModel();
		learnFeatures();
	}
	
	/**
	 * Update the parameter with given max iteration number
	 */
	private void learnFeatures(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate(ratings.getRandomIndex());
		}
	}
	
	/**
	 * In an iteration loop, update the user factors and item factors
	 * @param list the randomly generated index list
	 */
	private void iterate(ArrayList<Integer> list){
		
		int user_id, item_id, rating;
		double err, score, sig_score, prediction, gradient;
		Algebra algebra = new Algebra();
		for(int index = 0; index != trainNumber; ++index){
			
			user_id = ratings.getUser(index);
			item_id = ratings.getItem(index);
			rating  = ratings.getRating(index);
			
			score = globalBias + userBias[user_id] + itemBias[item_id]
			      + algebra.mult(userFeatures.viewRow(user_id), itemFeatures.viewRow(item_id));
			sig_score = 1 / (1 + Math.exp(-score));
			prediction = minRating + sig_score * ratingRange;
			err = prediction - rating;
			gradient = err * sig_score * ( 1 - sig_score ) * ratingRange;
			
			userBias[user_id] -= biasLearnRate * learnRate * (gradient + biasUserReg  * userBias[user_id]);
			itemBias[item_id] -= biasLearnRate * learnRate * (gradient + biasItemReg  * itemBias[item_id]);
			
			// update factors
			for(int f = 0;  f != featureNumber; ++f){
				double u_f = userFeatures.getQuick(user_id, f);
				double i_f = itemFeatures.getQuick(item_id, f);
				
				double delta_u = gradient * i_f + userReg * u_f;
				userFeatures.setQuick(user_id, f, userFeatures.getQuick(user_id, f) - learnRate * delta_u);
				
				double delta_i = gradient * u_f + itemReg * i_f;
				itemFeatures.setQuick(item_id, f, itemFeatures.getQuick(item_id, f) - learnRate * delta_i);
			}
		}
	}
	
	/**
	 * Predict the rating value with given user_id and item_id
	 */
	public double predict(int user_id, int item_id, boolean bound){
		
		if(user_id >= userFeatures.rows())
			return globalAvg;
		if(item_id >= itemFeatures.rows())
			return globalAvg;
		
		Algebra algebra = new Algebra();
		double result = userBias[user_id] + itemBias[item_id];
		result += globalBias;
		result += algebra.mult(userFeatures.viewRow(user_id), itemFeatures.viewRow(item_id));
		
		result =  (minRating + ( 1 / (1 + Math.exp(-result)) ) * ratingRange);
		
		if(bound){
			if( result > maxRating)
				result = (double) maxRating;
			if( result < minRating)
				result = (double) minRating;
		}
		//System.out.println(result);
		return result;
	}
}
