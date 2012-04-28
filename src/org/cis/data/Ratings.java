package org.cis.data;

import java.util.ArrayList;

public class Ratings implements DataSet {
	 
	//the number of interaction events in the data set
	int count;
	
	ArrayList<Integer> users;
	ArrayList<Integer> items;
	ArrayList<Integer> values;
	
	int maxUserId;
	int maxItemId;
	
	int maxRating;
	int minRating;
	
	ArrayList<ArrayList<Integer>> indexByUser;
	ArrayList<ArrayList<Integer>> indexByItem;
	
	ArrayList<Integer> randomIndex;
	
	public Ratings(){
		this.count  = 0;
		this.users  = new ArrayList<Integer>();
		this.items  = new ArrayList<Integer>();
		this.values = new ArrayList<Integer>();
		
		this.maxItemId = -1;
		this.maxUserId = -1;
		
		this.maxRating = Integer.MIN_VALUE;
		this.minRating = Integer.MAX_VALUE;
	}
	
	public void addRating(int user_id, int item_id, int rating){
		users.add(user_id);
		items.add(item_id);
		values.add(rating);
		
		count++;
		
		if(user_id > maxUserId)
			maxUserId = user_id;
		if(item_id > maxItemId)
			maxItemId = item_id;
		if(rating < minRating)
			minRating = rating;
		if(rating > maxRating)
			maxRating = rating;
		
	}
	
	@Override
	public void BuildUserIndices() {
		indexByUser = new ArrayList<ArrayList<Integer>>();
		for (int u = 0; u <= maxUserId; u++)
			indexByUser.add(new ArrayList<Integer>());
		// one pass over the data
		for (int index = 0; index < count; index++)
			indexByUser.get(users.get(index)).add(index);
	}
	
	@Override
	public void BuildItemIndices() {
		indexByItem = new ArrayList<ArrayList<Integer>>();
		for(int i = 0; i <= maxItemId; ++i)
			indexByItem.add(new ArrayList<Integer>());
		
		// ons pass over the data
		for (int index = 0; index < count; index++)
			indexByItem.get(items.get(index)).add(index);
	}
	
	@Override
	public void BuildRandomIndex() {
		randomIndex = new ArrayList<Integer>(count);
		for(int index = 0; index != count; ++index){
			randomIndex.add(index);
		}
		java.util.Collections.shuffle(randomIndex);
	}
	
	@Override
	public int GetIndex(int user_id, int item_id) {
		for(int index = 0; index != count; ++index)
			if(users.get(index) == user_id && items.get(index) == item_id)
				return index;
		return -1;
	}
	
	public ArrayList<ArrayList<Integer>> getIndicesByUser(){
		BuildUserIndices();
		return this.indexByUser;
	}
	
	public ArrayList<ArrayList<Integer>> getIndicesByItem(){
		BuildItemIndices();
		return this.indexByItem;
	}
	
	public ArrayList<Integer> getItemsByUser(int user_id){
		ArrayList<Integer> list = new ArrayList<Integer>();
		for( int i = 0; i != count; ++i){
			if(users.get(i) == user_id){
				list.add(items.get(i));
			}
		}
		return list;
	}
	
	public ArrayList<Integer> getUsersByItem(int item_id){
		ArrayList<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i != count; ++i){
			if(items.get(i) == item_id){
				list.add(users.get(i));
			}
		}
		return list;
	}
	
	public double averageRating(){
		double avg = 0;
		for( int i = 0; i != count; ++i)
			avg += values.get(i);
		return avg/count;
	}
	
	public int totalUserNumber(){
		return maxUserId;
	}
	
	public int totalItemNumber(){
		return maxItemId;
	}
	
	public int getMaxRating(){
		return maxRating;
	}
	
	public int getMinRating(){
		return minRating;
	}
	
	public int getCount(){
		return count;
	}
	
	public ArrayList<Integer> getRandomIndex(){
		BuildRandomIndex();
		return randomIndex;
	}
	
	public int getUser(int index){
		return users.get(index);
	}
	
	public int getItem(int index){
		return items.get(index);
	}
	
	public int getRating(int index){
		return values.get(index);
	}
	
	public void clear(){
		users.clear();
		items.clear();
		values.clear();
	}
}
