/*
 * Name: Angitha Mathew

 * K means clustering for Tweets
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.*;

//New class added to store the tweet, it's label and the prediction by the K means cluster. This will make the sorting and the printing easier.
//implements serializable to enable parallelize and convert it into a javaRDD
class TweetDetails implements java.io.Serializable {
    //attribute description in the constructor method
    int label;
    String tweet;
    int clusterID;

    /**
     * @param label     stores the label indicating spam or not
     * @param tweet     Stores the tweet that we're clustering
     * @param clusterID the clustername given by the prediction function. The value will be -1 before prediction
     *                  Constructor that'll initialize the value
     */
    public TweetDetails(int label, String tweet, int clusterID) {
        this.label = label;
        this.tweet = tweet;
        this.clusterID = clusterID;
    }

    /**
     * @return the label indicating spam or not
     */
    public int getLabel() {
        return label;
    }

    /**
     * @param label set's the label from the input text file
     */
    public void setLabel(int label) {
        this.label = label;
    }

    /**
     * @return the tweet that we're clustering
     */
    public String getTweet() {
        return tweet;
    }

    /**
     * @param tweet sets the tweet from the input file
     */
    public void setTweet(String tweet) {
        this.tweet = tweet;
    }

    /**
     * @return returns the value of the cluster the tweet belongs to
     */

    public int getclusterID() {
        return clusterID;
    }

    /**
     * @param clusterID initially set to -1, the value is updated once the predict function is called after we train the cluster
     */
    public void setclusterID(int clusterID) {
        this.clusterID = clusterID;
    }

    /**
     * @return just for debugging
     */
    @Override
    public String toString() {
        return "TweetDetails{" +
                "label=" + label +
                ", tweet='" + tweet + '\'' +
                ", clusterID=" + clusterID +
                '}';
    }
}

public class KMeansCluster {

    public static void main(String[] Args) {
        System.setProperty("hadoop.home.dir", "C:/winutils");//so that we can avoid setting the environment variable HADOOP_HOME manually
        SparkConf sparkConf = new SparkConf().setAppName("KMeansCluster")//sets the spark configuration with default values
                // expect for the ones manually set like AppName
                .setMaster("local[4]").set("spark.executor.memory", "1g");

        JavaSparkContext jsc = new JavaSparkContext(sparkConf);//JavaSparkContext object created: this is Java friendly and returns Java RDDs
        // Load and parse data
        //PLEASE CHANGE THIS PATH BEFORE TRYING TO RUN!!
        String path = "D:\\MAI-SEMESTER-1\\LSDA\\Assignment4\\twitter2D_2.txt";
        JavaRDD<String> data = jsc.textFile(path);
        // Please note that parts of the below code is taken from the apache spark official documentation as suggested in the assignment
        //The data read from the text file is split into array of strings based on commas as it is a comma separated list
        //the 3rd and the forth strings respectively indicating the co-ordinates of the location of the tweeter, is taken separately so that
        //it can be passed to the train function as mentioned in the assignment.
        JavaRDD<Vector> X_train = data.map(s -> {
            String[] sarray = s.split(",");
            double[] values = new double[2];
            values[0] = Double.parseDouble(sarray[2]);
            values[1] = Double.parseDouble(sarray[3]);
            return Vectors.dense(values);
        });
        // The label indicating the spam, the tweet and the cluster Id - default value -1 is passed to the constructor of the class Tweet Details
        //so that it can all be stored and accessed easily
        JavaRDD<TweetDetails> Tweet = data.map(s -> {
            String[] sarray = s.split(",");
            TweetDetails storedValue = new TweetDetails(Integer.valueOf(sarray[0]), sarray[1], -1);//initialing all predicts with a random value -1
            return storedValue;
        });
        X_train.cache();
        // Cluster the data into two classes using KMeans
        int numClusters = 4; // number of clusters in the trained Model
        int[] spamCount = new int[numClusters]; //An array of integers to keep track of the spam count
        int numIterations = 20;// Can be modified according to the training data and the value of the evaluation metrics.
        //Creating the KMeansModel object class with the training data, the number of clusters needed and the num of iterations.
        KMeansModel clusters = KMeans.train(X_train.rdd(), numClusters, numIterations);
        //Once the model is trained, the X_train (which is both the training and testing set in this case) is passed and predictions -
        //an integer denoting the cluster ID is returned.
        JavaRDD<Integer> predictions = clusters.predict(X_train);
        //Not required. But the cluster centers for the current classes of the cluster are printed
        System.out.println("Cluster centers:");// Printed Just for the info, feel free to ignore it.
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        //Updating the tweet detail with the predictions we got from the trained model
        List<TweetDetails> allTweets = Tweet.collect();
        List<Integer> ListPredictions = predictions.collect();
        //iterating over the list of predictions and the list of tweetDetails and updating the values
        //Since the values are not shuffled in anyway, they'll be the corresponding value
        for (int i = 0; i < ListPredictions.size(); i++) {
            allTweets.get(i).setclusterID(ListPredictions.get(i));
        }
        //Once we have all the information required in a list of object, we sort the object based on the attribute given- in this case the cluster ID
        List<TweetDetails> allTweetssorted = jsc.parallelize(allTweets).sortBy(s -> s.clusterID, true, 1).collect();
        //Requirement 1 for the assignment
        //Printing the Tweet with the corresponding CLuster Id as mentioned.
        for (TweetDetails eachTweet : allTweetssorted) {
            System.out.println("Tweet " + eachTweet.getTweet() + " is in cluster " + eachTweet.getclusterID());
            // ALso, making note of the frequency of a spam in each of the Clusters and storing in a List
            if (eachTweet.label == 1) {
                spamCount[eachTweet.clusterID]++;
            }
        }
        //Printing the spam count for each clusters as required in the assignment
        //Requirement 2 for the assignment
        for (int j = 0; j < numClusters; j++) {
            System.out.println("Cluster " + j + " contains " + spamCount[j] + " spam tweet");

        }

        jsc.stop();
        jsc.close();//Stopping and closing the spark context ensuring all the variables are cleaned up.
    }
}
