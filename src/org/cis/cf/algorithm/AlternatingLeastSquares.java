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

public class AlternatingLeastSquares implements RatingPredictor{
	Ratings ratings;

	SparseDoubleMatrix2D trainMatrix;
	SparseDoubleMatrix2D logitMatrix;
	
	DenseDoubleMatrix2D userFeatures;
	DenseDoubleMatrix2D itemFeatures;
	
	DoubleMatrix2D subUserFeatures;
	DoubleMatrix2D subItemFeatures;
	
	DoubleMatrix1D Ri;
	DoubleMatrix1D Rj;
	
	DoubleMatrix2D E;
	
	DenseDoubleMatrix1D userRateNumber;
	DenseDoubleMatrix1D itemRatedNumber;
	
	int userNumber;
	int itemNumber;
	
	int featureNumber;
	
	double userReg;
	double itemReg;
	
	int maxIterNumber;
	
	double globalBias;
	int maxRating;
	int minRating;
	
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
	
	private void convertData(){
		int count = ratings.getCount();
		for( int index = 0; index != count; ++index){
			trainMatrix.setQuick(ratings.getUser(index), ratings.getItem(index), ratings.getRating(index));
			logitMatrix.setQuick(ratings.getUser(index), ratings.getItem(index), 1);
		}
		ratings.clear();
	}
	
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
	
	private void learnFeatures(){
		for(int iter = 1; iter <= maxIterNumber; ++iter){
			iterate();
		}
	}
	
	private void iterate(){
		genU();
		genM();
	}
	
	public void trainModel() {
		initModel();
		learnFeatures();
		
	}

	@Override
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

	public void genRi(int i) {
		Ri = trainMatrix.viewRow(i).viewSelection(
				new DoubleProcedure() {
					public final boolean apply(double element) {
						return element != 0;
					}
				});
	}


	public void genRj(int j) {
		Rj = trainMatrix.viewColumn(j).viewSelection(
				new DoubleProcedure() {
					public final boolean apply(double element) {
						return element != 0;
					}
				});
	}

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
