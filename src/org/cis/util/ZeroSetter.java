package org.cis.util;

public class ZeroSetter {
	
	public static void zero(double[] a, int l1) {
		for(int i = 0; i < l1; i++)
				a[i] = 0;
	}
	
	public static void zero(int[] a, int l1) {
		for(int i = 0; i < l1; i++)
				a[i] = 0;
	}
	
	public static void zero(int[][] a, int l1, int l2) {
		for(int i = 0; i < l1; i++)
			for(int j = 0; j < l2; j++)
				a[i][j] = 0;
	}
	
	public static void zero(double[][] a, int l1, int l2) {
		for(int i = 0; i < l1; i++)
			for(int j = 0; j < l2; j++)
				a[i][j] = 0.0;
	}
	
	public static void zero(double[][][] a, int l1, int l2, int l3) {
		for(int i = 0; i < l1; i++)
			for(int j = 0; j < l2; j++)
				for(int k = 0; k < l3; k++)
					a[i][j][k] = 0.0;
	}
}
