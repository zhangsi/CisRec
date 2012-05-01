package org.cis.cf.algorithm;

import java.util.Random;

import org.cis.data.Ratings;


import cern.colt.function.DoubleProcedure;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * The class implementing the Alternating Least Squares algorithm
 * 
 * The origin paper:
 * 
 * Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan. 
 * Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
 * Proceedings of the 4th international conference on Algorithmic Aspects in Information and Management.
 * Shanghai, China pp. 337-348, 2008.
 * http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf
 *
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class AlternatingLeastSquares implements RatingPredictor{
	
	/** training data set of ratings */
	Ratings ratings;

	/** training data set represented by sparse matrix */
	SparseDoubleMatrix2D trainMatrix;
	/** indicator of training sparse matrix */
	SparseDoubleMatrix2D logitMatrix;
	
	/** user factors */
	DenseDoubleMatrix2D userFeatures;
	/** item factors */
	DenseDoubleMatrix2D itemFeatures;
	
	/** sub user factor */
	DoubleMatrix2D subUserFeatures;
	/** sub item factor */
	DoubleMatrix2D subItemFeatures;
	
	/** vector of a user */
	DoubleMatrix1D Ri;
	/** vector of an item */
	DoubleMatrix1D Rj;
	
	/** eye matrix */
	DoubleMatrix2D E;
	
	/** the ratings number involved with the user */
	DenseDoubleMatrix1D userRateNumber;
	/** the ratings number involved with the item */
	DenseDoubleMatrix1D itemRatedNumber;
	
	/** number of users */
	int userNumber;
	/** number of items */
	int itemNumber;
	
	/** number of latent features */
	int featureNumber;
	
	/** regularization of user factors */
	double userReg;
	/** regularization of item factors */
	double itemReg;
	
	/** max iteration number */
	int maxIterNumber;
	
	/** global rating average */ 
	double globalBias;
	
	/** max rating */
	int maxRating;
	/** min rating */
	int minRating;
	
	/**
	 * Construct ALS algorithm 
	 * 
	 * @param ratings
	 * @param featureNumber
	 * @param userReg
	 * @param itemReg
	 * @param maxIterNumber
	 */
	public AlternatingLeastSquares(Ratings ratings, int featureNumber,
			double userReg, double itemReg, int maxIterNumber) {
		this.ratings = ratings;
		this.userNumber = ratings.totalUserNumber();
		this.itemNumber = ratings.totalItemNumber();
		
		this.featureNumber = featureNumber;
		this.userReg = userReg;
		this.itemReg = itemReg;
		
		this.globalBias = ratings.averageRating();
		this.maxRating  = ratings.getMaxRating();
		this.minRating  = ratings.getMinRating();
		
		this.userFeatures = new DenseDoubleMatrix2D(featureNumber, userNumber + 1);
		this.itemFeatures = new DenseDoubleMatrix2D(featureNumber, itemNumber + 1);
		this.trainMatrix = new SparseDoubleMatrix2D(userNumber + 1, itemNumber + 1);
		this.logitMatrix = new SparseDoubleMatrix2D(userNumber + 1, itemNumber + 1);
		
		this.userRateNumber  = new DenseDoubleMatrix1D( userNumber + 1);
		this.itemRatedNumber = new DenseDoubleMatrix1D( itemNumber + 1);
		
		E = DoubleFactory2D.sparse.identity(featureNumber);
		
		this.maxIterNumber = maxIterNumber;
		
		convertData();
	}
	
	/**
	 * Convert training data from Ratings to sparse matrix
	 */
	private void convertData(){
		int count = ratings.getCount();
		for( int index = 0; index != count; ++index){
			trainMatrix.setQuick(ratings.getUser(index), ratings.getItem(index), ratings.getRating(index));
			logitMatrix.setQuick(ratings.getUser(index), ratings.getItem(index), 1);
		}
		ratings.clear();
	}
	
	/**
	 * Init model parameters
	 */
	private void initModel(){
		Random rand = new Random();
		rand.setSeed(0);
		for( int u = 1; u <= userNumber; ++u){
			for( int f = 0; f != featureNumber; ++f){
				userFeatures.setQuick(f, u, rand.nextGaussian() * 0.01);
			}
			userRateNumber.setQuick(u, logitMatrix.viewRow(u).cardinality());
		}
		for( int i = 1; i <= itemNumber; ++i){
			for( int f = 0; f != featureNumber; ++f){
				itemFeatures.setQuick(f, i, rand.nextGaussian() * 0.01);
			}
			itemRatedNumber.setQuick(i, logitMatrix.viewColumn(i).cardinality());
		}
	}
	
	/**
	 * Learn the user and item factors with given max iteration number
	 */
	private void learnFeatures(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate();
		}
	}
	
	/**
	 * Update user factors and item factors in each iteration
	 */
	private void iterate(){
		genU();
		genM();
	}
	
	/**
	 * Train the ALS model
	 */
	public void trainModel() {
		initModel();
		learnFeatures();
		
	}

	/**
	 * Predict the rating value with given user and item
	 */
	public double predict(int user_id, int item_id, boolean bound) {
		
		Algebra algebra = new Algebra();
		double result = 0;

		result = algebra.mult(userFeatures.viewColumn(user_id), itemFeatures.viewColumn(item_id));
		
		if(bound){
			if( result > maxRating)
				result = (double) maxRating;
			if( result < minRating)
				result = (double) minRating;
		}
		return result;
	}
	
	/**
	 * Generate the sub item factors involved with the given user
	 * @param i the given user id
	 */
	public void gensubItemFeatures(int i) {
		IntArrayList indexList = new IntArrayList();
		DoubleArrayList valueList = new DoubleArrayList();
		logitMatrix.viewRow(i).getNonZeros(indexList, valueList);
		int[] rowIndexes = new int[featureNumber];
		int l = indexList.size();
		int[] columnIndexes = new int[l];
		int x;
		for (x = 0; x != l; ++x) {
			columnIndexes[x] = indexList.get(x);
		}
		for (x = 0; x != featureNumber; ++x) {
			rowIndexes[x] = x;
		}
		subItemFeatures = itemFeatures.viewSelection(rowIndexes, columnIndexes);
	}

	/**
	 * Generate the sub user factors involved with the given item
	 * @param j the given item id
	 */
	public void gensubUserFeatures(int j) {
		IntArrayList indexList = new IntArrayList();
		DoubleArrayList valueList = new DoubleArrayList();
		logitMatrix.viewColumn(j).getNonZeros(indexList, valueList);
		int[] rowIndexes = new int[featureNumber];
		int x;
		int l = indexList.size();
		int[] columnIndexes = new int[l];
		for (x = 0; x != l; ++x) {
			columnIndexes[x] = indexList.get(x);
		}
		for (x = 0; x != featureNumber; ++x) {
			rowIndexes[x] = x;
		}
		subUserFeatures = userFeatures.viewSelection(rowIndexes, columnIndexes);
	}

	/**
	 * Generate the vector contains the items involved with the given user
	 * @param i the user id
	 */
	public void genRi(int i) {
		Ri = trainMatrix.viewRow(i).viewSelection(
				new DoubleProcedure() {
					public final boolean apply(double element) {
						return element != 0;
					}
				});
	}

	/**
	 * Generate the vector contains the users involved with the give item
	 * @param j the item id
	 */
	public void genRj(int j) {
		Rj = trainMatrix.viewColumn(j).viewSelection(
				new DoubleProcedure() {
					public final boolean apply(double element) {
						return element != 0;
					}
				});
	}

	/**
	 * Update the factor of given user
	 * @param i the user id
	 * @return the updated user factor
	 */
	public DoubleMatrix1D genUi(int i) {
		Algebra algebra = new Algebra(0);
		Algebra algebra2 = new Algebra(0);
		gensubItemFeatures(i);
		genRi(i);
		E = DoubleFactory2D.sparse.identity(featureNumber);
		return algebra
				.mult(
				// inverse of Ai
						algebra2.inverse(subItemFeatures.zMult(subItemFeatures.viewDice(), E,
								1, userReg * userRateNumber.getQuick(i), false,
								false)),
						// Vi
						algebra.mult(subItemFeatures, Ri));
	}

	/**
	 * Update the factor of given item
	 * @param j the item id
	 * @return the updated item factor
	 */
	public DoubleMatrix1D genMj(int j) {
		if (logitMatrix.viewColumn(j).cardinality() == 0)
			return DoubleFactory1D.sparse.make(featureNumber, 0);
		Algebra algebra = new Algebra(0);
		gensubUserFeatures(j);
		genRj(j);
		E = DoubleFactory2D.sparse.identity(featureNumber);
		return algebra.mult(
		// inverse of Aj
				algebra.inverse(subUserFeatures.zMult(subUserFeatures.viewDice(), E, 1,
						itemReg * itemRatedNumber.getQuick(j), false, false)),
				// Vj
				algebra.mult(subUserFeatures, Rj));
	}

	/**
	 * update user factors
	 */
	public void genU() {
		int i, j;
		DoubleMatrix1D Ui;
		for (i = 1; i <= userNumber; ++i) {
			// System.out.print("the " + i + "th user:");
			Ui = genUi(i);
			for (j = 0; j != featureNumber; ++j) {
				userFeatures.setQuick(j, i, Ui.getQuick(j));
			}
		}
	}

	/**
	 * update item factors
	 */
	public void genM() {
		int i, j;
		DoubleMatrix1D Mj;
		for (j = 1; j <= itemNumber; ++j) {
			Mj = genMj(j);
			for (i = 0; i != featureNumber; ++i) {
				itemFeatures.setQuick(i, j, Mj.getQuick(i));
			}
		}
	}
}
