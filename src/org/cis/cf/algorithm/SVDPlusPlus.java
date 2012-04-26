package org.cis.cf.algorithm;

import java.util.ArrayList;
import java.util.Random;

import org.cis.data.Ratings;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

public class SVDPlusPlus implements RatingPredictor{
	
	DenseDoubleMatrix2D userFeatures;
	DenseDoubleMatrix2D itemFeatures;
	DenseDoubleMatrix2D p;
	DenseDoubleMatrix2D y;
	Ratings ratings;
	int maxRating;
	int minRating;
	
	int trainNumber;
	
	double globalBias;
	
	double learnRate;
	double userReg;
	double itemReg;
	
	double userBias[];
	double itemBias[];
	
	double biasLearnRate;
	double biasUserReg;
	double biasItemReg;
	
	int featureNumber;
	int maxIterNumber;
	
	int userNumber;
	int itemNumber;
	
	int whoRatedWhat[][];
	
	public SVDPlusPlus(Ratings ratings, int featureNumber,
			double learnRate, double userReg, double itemReg, 
			double biasLearnRate, double biasUserReg, double biasItemReg,
			int maxIterNumber) {
		this.ratings = ratings;
		
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.maxRating = ratings.getMaxRating();
		this.minRating = ratings.getMinRating();
		
		this.globalBias = ratings.averageRating();
		
		this.trainNumber = ratings.getCount();
		
		this.learnRate = learnRate;
		this.userReg   = userReg;
		this.itemReg   = itemReg;
		
		this.biasItemReg = biasItemReg;
		this.biasUserReg = biasUserReg;
		this.biasLearnRate = biasLearnRate;
		
		
		this.maxIterNumber = maxIterNumber;
		
		this.featureNumber = featureNumber;
		
		this.p = new DenseDoubleMatrix2D(userNumber + 1, featureNumber);
		this.y = new DenseDoubleMatrix2D(itemNumber + 1, featureNumber);
		this.userFeatures = new DenseDoubleMatrix2D(userNumber + 1, featureNumber);
		this.itemFeatures = new DenseDoubleMatrix2D(itemNumber + 1, featureNumber);
		
		this.userBias     = new double[userNumber + 1];
		this.itemBias     = new double[itemNumber + 1];
	}
	
	private void getImplicitInfo(){
		whoRatedWhat = new int[userNumber+1][];
		ArrayList<Integer> list = new ArrayList<Integer>();
		for(int u = 1; u <= userNumber; ++u){
			list = ratings.getItemsByUser(u);
			int size = list.size();
			whoRatedWhat[u] = new int[size];
			for( int i = 0; i != size; ++i)
				whoRatedWhat[u][i] = list.get(i);
		}
	}
	
	private void initModel(){
		Random rand = new Random();
		
		for( int u = 0; u != userNumber; ++u){
			for( int f = 0; f != featureNumber; ++f){
				p.setQuick(u, f, rand.nextGaussian() * 0.01);
			}
			userBias[u] = rand.nextGaussian() * 0.01;
		}
		
		for( int i = 0; i != itemNumber; ++i){
			for( int f = 0; f != featureNumber; ++f){
				itemFeatures.setQuick(i, f, rand.nextGaussian() * 0.01);
				y.setQuick(i, f, rand.nextGaussian() * 0.01);
			}
			itemBias[i] = rand.nextGaussian() * 0.01;
		}
	}
	
	public void trainModel(){
		initModel();
		getImplicitInfo();
		learnFeatures();
		calcUserFeatures();
	}
	
	private void learnFeatures(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate(ratings.getRandomIndex());
		}
	}

	private void iterate(ArrayList<Integer> list){
		int user_id, item_id, rating;
		double err, prediction;
		Algebra algebra = new Algebra();
		for(int index : list){
			
			user_id = ratings.getUser(index);
			item_id = ratings.getItem(index);
			rating  = ratings.getRating(index);
			
			prediction = globalBias + userBias[user_id] + itemBias[item_id];
			int len = whoRatedWhat[user_id].length;
			double norm_denominator = Math.sqrt(len);
			DenseDoubleMatrix1D userPlusY = new DenseDoubleMatrix1D(featureNumber);
			userPlusY.assign(0);
			for( int j = 0; j != len; ++j){
				for( int d = 0; d != featureNumber; ++d){
					userPlusY.setQuick(d, userPlusY.getQuick(d) + y.getQuick(whoRatedWhat[user_id][j], d));
				}
			}
			for( int d = 0; d != featureNumber; ++d){
				userPlusY.setQuick(d, userPlusY.getQuick(d)/norm_denominator + p.getQuick(user_id, d));
			}
			
			prediction += algebra.mult(userPlusY, itemFeatures.viewRow(item_id));
			err = rating - prediction;
			
			userBias[user_id] += biasLearnRate * learnRate * (err - biasUserReg * userBias[user_id]);
			itemBias[item_id] += biasLearnRate * learnRate * (err - biasItemReg * itemBias[item_id]);
			
			double x = err / norm_denominator;
			for(int d = 0; d != featureNumber; ++d){
				double i_f = itemFeatures.getQuick(item_id, d);
				double delta_u = err * i_f - userReg * p.getQuick(user_id, d);
				p.setQuick(user_id, d, p.getQuick(user_id, d) + learnRate * delta_u);
				
				double delta_i = err * userPlusY.getQuick(d) - itemReg * i_f;
				itemFeatures.setQuick(item_id, d, itemFeatures.getQuick(item_id, d) + learnRate * delta_i);
				
				double common_update = x * i_f;
				for( int j = 0; j != len; ++j){
					double delta_oi = common_update - itemReg * y.getQuick(whoRatedWhat[user_id][j], d);
					y.setQuick(whoRatedWhat[user_id][j], d, y.getQuick(whoRatedWhat[user_id][j], d) +  + learnRate * delta_oi);
				}
			}
		}
	}
	
	private void calcUserFeatures(){
		int user_id;
		for(user_id = 1; user_id <= userNumber; ++user_id){
			int len = whoRatedWhat[user_id].length;
			double norm_denominator = Math.sqrt(len);
			DenseDoubleMatrix1D userPlusY = new DenseDoubleMatrix1D(featureNumber);
			userPlusY.assign(0);
			for( int j = 0; j != len; ++j){
				for( int d = 0; d != featureNumber; ++d){
					userPlusY.setQuick(d, userPlusY.getQuick(d) + y.getQuick(whoRatedWhat[user_id][j], d));
				}
			}
			for( int d = 0; d != featureNumber; ++d){
				userPlusY.setQuick(d, userPlusY.getQuick(d)/norm_denominator + p.getQuick(user_id, d));
				userFeatures.setQuick(user_id, d, userPlusY.getQuick(d));
			}
		}
	}

	@Override
	public double predict(int user_id, int item_id, boolean bound){
		
		if(user_id >= p.rows())
			return globalBias;
		if(item_id >= itemFeatures.rows())
			return globalBias;
		
		Algebra algebra = new Algebra();
		double result = userBias[user_id] + itemBias[item_id];
		result += globalBias;
		result += algebra.mult(p.viewRow(user_id), itemFeatures.viewRow(item_id));
		
		if(bound){
			if( result > maxRating)
				result = (double) maxRating;
			if( result < minRating)
				result = (double) minRating;
		}
		return result;
	}
}
