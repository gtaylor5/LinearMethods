import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class AdalineNN {
    
    AdalineNeuron[] neurons;
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    ArrayList<int[]> testData = new ArrayList<int[]>();
    ArrayList<int[]> validationData = new ArrayList<int[]>();
    HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
    String dataSetName = "";
    
    
    
    /************************************************************
    Returns performance of perceptron.
    ************************************************************/
    
    public double testAdaline(){
        double performance = 0;
        for(int[] val : testData){
            for(int i = 0; i < neurons.length; i++){
                if(neurons[i].test(val) == 1){
                    performance++;
                }
            }
        }
        return performance*100/((double)testData.size());
    }
    
    /************************************************************
    For each of the neurons in the perceptron train them based on
    their class value.
     * @throws IOException 
    ************************************************************/
    
    public void trainAdaline() throws IOException{
        for(int i = 0; i < neurons.length; i++){
            neurons[i].train();
        }
    }
    
    /************************************************************
    Initialize Array of Neurons with class value and training and
    validation data
    ************************************************************/
    
    public void setNeurons(){
        neurons = new AdalineNeuron[classCounts.size()];
        int i = 0;
        for(Integer key: classCounts.keySet()){
            neurons[i] = new AdalineNeuron(key, trainingData, validationData);
            i++;
        }
    }
    
    public void printWeights(){
        for(AdalineNeuron n : neurons){
            Main.writer2.print("Weights for Class " + n.classVal + " : ");
            for(double w : n.weights){
                Main.writer2.print(w + " ");
            }
            Main.writer2.println();
            Main.writer2.println();
        }
    }
    
    
    
    /************************************************************
    Counts all of the classes in the training set. 
    ************************************************************/
    
    public void countClasses(){
        //handle special cases.
        for(int i = 0; i < trainingData.size(); i++){
            if(classCounts.containsKey(trainingData.get(i)[trainingData.get(i).length-1])){ // hashmap contains class?
                int currentVal = classCounts.get(trainingData.get(i)[trainingData.get(i).length-1]); // get current count
                currentVal++; // increment count by 1
                classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], currentVal); //update map.
            }else{
                classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], 1); // add class to map
            }
        }
    }
    
    /************************************************************
    Fills the training file based on the index that is to be skipped.
    Used specifically for cross validation. Also, this method
    inserts an n+1 element equal to -1 while maintaining the class
    label location to be at length-1.
    ************************************************************/
    
    void fillTrainFile(int indexToSkip) throws IOException{
        for(int i = 1; i < 6; i++){
            if(i == indexToSkip) continue; // skip file at index.
            Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+i+".txt"));
            while(fileScanner.hasNextLine()){
                String[] arr = fileScanner.nextLine().split(" ");

                int[] vals = new int[arr.length + 1];
                for(int j = 0; j < vals.length; j++){
                    if(j == arr.length-1){
                        vals[j+1] = Integer.parseInt(arr[j]);
                        vals[j] = -1;
                        break;
                    }
                    vals[j] = Integer.parseInt(arr[j]);
                }
                trainingData.add(vals);
            }
            fileScanner.close();
        }
    }
    
    /************************************************************
    Fills test file similarly to fillTrainFile.
    ************************************************************/
    
    void fillTestFile(int indexToSkip) throws IOException{
        Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+indexToSkip+".txt"));
        while(fileScanner.hasNextLine()){
            String[] arr = fileScanner.nextLine().split(" ");
            int[] vals = new int[arr.length+1];
            for(int j = 0; j < arr.length; j++){
                if(j == arr.length-1){
                    vals[j+1] = Integer.parseInt(arr[j]);
                    vals[j] = -1;
                    break;
                }
                vals[j] = Integer.parseInt(arr[j]);
            }
            testData.add(vals);
        }
        fileScanner.close();
    }
    
    /************************************************************
    Fills the validation set.
    ************************************************************/
    void fillValidationSet() throws IOException{
        Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/validationSet.txt"));
        while(fileScanner.hasNextLine()){
            String[] arr = fileScanner.nextLine().split(" ");

            int[] vals = new int[arr.length+1];
            for(int j = 0; j < arr.length; j++){
                if(j == arr.length-1){
                    vals[j+1] = Integer.parseInt(arr[j]);
                    vals[j] = -1;
                    break;
                }
                vals[j] = Integer.parseInt(arr[j]);
            }
            validationData.add(vals);
        }
        fileScanner.close();
    }

}
