package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;
import org.cis.matrix.SparseBooleanMatrix;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

public class SocialMatrixFactorization extends BiasedProbabilisticMatrixFactorization {

	private SparseBooleanMatrix user_connections;
	private SparseBooleanMatrix user_reverse_connections;
	private double socialReg;
	
	public SocialMatrixFactorization(Ratings ratings, int featureNumber, 
			SparseBooleanMatrix user_connections, double socialReg) {
		super(ratings, featureNumber);
		this.user_connections = user_connections;
		this.user_reverse_connections = user_connections.transpose();
		this.socialReg = socialReg;
	}
	public SocialMatrixFactorization(Ratings ratings, int featureNumber,
			double learnRate, double userReg, double itemReg, 
			double biasLearnRate, double biasUserReg, double biasItemReg,
			int maxIterNumber, SparseBooleanMatrix user_connections, double socialReg) {

		super(ratings, featureNumber, learnRate, userReg, itemReg, biasLearnRate,biasUserReg, biasItemReg, maxIterNumber);
		this.user_connections = user_connections;
		this.user_reverse_connections = user_connections.transpose();
		this.socialReg = socialReg;

	}
	
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
	
	private void learnFeatures(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate(ratings.getRandomIndex());
		}
	}
	
	public void trainModel(){
		initModel();
		learnFeatures();
	}
	
	
	private void iterate(ArrayList<Integer> list){
		DenseDoubleMatrix2D user_factors_gradient = new DenseDoubleMatrix2D( userNumber + 1, featureNumber);
		DenseDoubleMatrix2D item_factors_gradient = new DenseDoubleMatrix2D( itemNumber + 1, featureNumber);
		user_factors_gradient.assign(0);
		item_factors_gradient.assign(0);
			
		double [] user_bias_gradient    = new double[userNumber + 1];
		double [] item_bias_gradient    = new double[itemNumber + 1];
		for( int u = 0; u != user_bias_gradient.length; ++u)
			user_bias_gradient[u] = 0;
		for( int i = 0; i != item_bias_gradient.length; ++i)
			item_bias_gradient[i] = 0;
		
		
		int index;
		int user_id, item_id, rating;
		double err, score, sig_score, prediction, gradient;
		Algebra algebra = new Algebra();
		for(index = 0; index != trainNumber; ++index){
			
			user_id = ratings.getUser(index);
			item_id = ratings.getItem(index);
			rating  = ratings.getRating(index);
			
			score = globalBias + userBias[user_id] + itemBias[item_id]
			      + algebra.mult(userFeatures.viewRow(user_id), itemFeatures.viewRow(item_id));
			sig_score = 1 / (1 + Math.exp(-score));
			prediction = minRating + sig_score * ratingRange;
			err =  prediction - rating;
			gradient = err * sig_score * ( 1 - sig_score ) * ratingRange;

			user_bias_gradient[user_id] += gradient;
			item_bias_gradient[item_id] += gradient;
			
			for (int f = 0; f < featureNumber; f++){
				double u_f = userFeatures.get(user_id, f);
				double i_f = itemFeatures.get(item_id, f);

				user_factors_gradient.setQuick(user_id, f, user_factors_gradient.getQuick(user_id, f) + gradient * i_f);
				item_factors_gradient.setQuick(item_id, f, item_factors_gradient.getQuick(item_id, f) + gradient * u_f);
			}
		}
		
		// I.2 L2 regularization
		//        biases
		for (int u = 0; u < userNumber + 1; u++)
			user_bias_gradient[u] += userBias[u] * userReg * biasUserReg;
		for (int i = 0; i < itemNumber + 1; i++)
			item_bias_gradient[i] += itemBias[i] * itemReg * biasItemReg;
		//        latent factors
		for (int u = 0; u < userNumber + 1; u++)
			for (int f = 0; f < featureNumber; f++)
				user_factors_gradient.setQuick(u, f, user_factors_gradient.getQuick(u, f) + userFeatures.getQuick(u, f) * userReg);

		for(int i = 0; i < itemNumber + 1; i++)
			for(int f = 0;  f < featureNumber; f++)
				item_factors_gradient.setQuick(i, f, item_factors_gradient.getQuick(i, f) + itemFeatures.getQuick(i, f) * itemReg);

		// I.3 social network regularization -- see eq. (13) in the paper
			for (int u = 0; u < userNumber + 1; u++)
			{
				double [] sum_connections   = new double[featureNumber];
				double bias_sum_connections = 0;
				int num_connections         = user_connections.getNumEntriesByRow(u);
				
				for(int v : user_connections.getRow(u)){
					bias_sum_connections += userBias[v];
					for (int f = 0; f < featureNumber; f++)
						sum_connections[f] += userFeatures.getQuick(v, f);
				}
				
				if (num_connections != 0)
				{
					user_bias_gradient[u] += socialReg * (userBias[u] - bias_sum_connections / num_connections);
					for (int f = 0; f < featureNumber; f++)
						user_factors_gradient.setQuick(u, f, user_factors_gradient.getQuick(u, f)
								+ socialReg * (userFeatures.getQuick(u, f) - sum_connections[f] / num_connections));
				}
				
				for (int v : user_reverse_connections.getRow(u)){
					if (user_connections.getNumEntriesByRow(v) != 0){
						
						double trust_v = (float) 1 / user_connections.getNumEntriesByRow(v);
						double neg_trust_times_reg = - socialReg * trust_v;

						double bias_diff = 0;
						double[] factor_diffs = new double[featureNumber];
						for (int w : user_connections.getRow(v))
						{
							bias_diff -= userBias[w];
							for (int f = 0; f < featureNumber; f++)
								factor_diffs[f] -= userFeatures.getQuick(w, f);
						}
						
						bias_diff *= trust_v; // normalize
						bias_diff += userBias[v];
						user_bias_gradient[u] += neg_trust_times_reg * bias_diff;

						for (int f = 0; f < featureNumber; f++)
						{
							factor_diffs[f] *= trust_v; // normalize
							factor_diffs[f] += userFeatures.getQuick(v, f);
							user_factors_gradient.setQuick(u, f, user_factors_gradient.getQuick(u, f)
									+ neg_trust_times_reg * factor_diffs[f]);
						}
					}
				}

			}

		// II. apply gradient descent step
		for ( user_id = 0; user_id < userNumber + 1; user_id++)
			userBias[user_id] -= user_bias_gradient[user_id] * learnRate * biasLearnRate;
		for ( item_id = 0; item_id < itemNumber; item_id++)
			itemBias[item_id] -= item_bias_gradient[item_id] * learnRate * biasLearnRate;
		
		for( int u = 0; u != userNumber + 1; ++u){
			for(int f = 0; f != featureNumber; ++f){
				userFeatures.setQuick(u, f, userFeatures.getQuick(u, f) - learnRate * user_factors_gradient.getQuick(u, f));
			}
		}
		for( int i = 0; i != itemNumber + 1; ++i){
			for(int f = 0; f != featureNumber; ++f){
				itemFeatures.setQuick(i, f, itemFeatures.getQuick(i, f) - learnRate * item_factors_gradient.getQuick(i, f));
			}
		}
		
	}
	
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
