import java.io.File;
import java.io.IOException;
import java.util.*;

class Main{

      ArrayList<int[]> trainingData = new ArrayList<int[]>(); //training data as usable arraylist
    ArrayList<int[]> testData = new ArrayList<int[]>(); //test data as usable arraylist
    ArrayList<int[]> validationSet = new ArrayList<int[]>(); //validation set as usable arraylist
    String dataSetName; //the name of the dataset being tested/trained
  
  
    public Main(String name){
      this.dataSetName = name;
    }
  
  
    public static void main(String[] args) throws IOException{
      String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
      for(int i = 0; i < dataSets.length; i++){
          System.out.println(dataSets[i]);
        for(int j = 1; j < 6; j++){
          System.out.println("Fold #: "+j);
          Main m = new Main(dataSets[i]);
          m.fillTestFile(j);
          m.fillTrainFile(j);
          m.fillValidationSet();
          m.test();
          System.out.println();
        }
        System.out.println("-------------------");
      }
    }
  
  
    void test() throws IOException{
    
      //Logistic Learning
  
      LogisticLearner l = new LogisticLearner(this.dataSetName);
      l.eta = .001;
      l.trainingData = this.trainingData;
      l.testData = this.testData;
      l.validationSet = this.validationSet;
      l.countClasses();
      l.initializeClasses();
      l.gradientDescent();
      System.out.printf("Logistic : %.2f", l.test()*100);
      System.out.print(" %");
      System.out.println();
    
      //Naive Bayes
    
      NaiveBayes b = new NaiveBayes(this.dataSetName);
      b.trainingData = this.trainingData;
      b.testData = this.testData;
      b.countClasses();
      for(int key : b.classCounts.keySet()){
        b.trainNaiveBayes(b.trainingData, key);
      }
      b.testNaiveBayes(testData);
    
      //Perceptron Learning 
    
      Perceptron p = new Perceptron();
      p.dataSetName = this.dataSetName;
      p.trainingData = this.trainingData;
      p.testData = this.testData;
      p.validationData = this.validationSet;
      p.countClasses();
      p.setNeurons();
      p.trainPerceptron();
      p.testPerceptron();
    
      //Adaline Learning
    
      AdalineNN a = new AdalineNN();
      a.dataSetName = this.dataSetName;
      a.trainingData = this.trainingData;
      a.testData = this.testData;
      a.validationData = this.validationSet;
      a.countClasses();
      a.setNeurons();
      a.trainAdaline();
      a.testAdaline();
    }

    /************************************************************
    Fills the training file based on the index that is to be skipped.
    Used specifically for cross validation.
    ************************************************************/
    
    void fillTrainFile(int indexToSkip) throws IOException{
        for(int i = 1; i < 6; i++){
            if(i == indexToSkip) continue; // skip file at index.
            Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+i+".txt"));
            while(fileScanner.hasNextLine()){
                String[] arr = fileScanner.nextLine().split(" ");

                int[] vals = new int[arr.length];
                for(int j = 0; j < arr.length; j++){
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
            int[] vals = new int[arr.length];
            for(int j = 0; j < arr.length; j++){
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

            int[] vals = new int[arr.length];
            for(int j = 0; j < arr.length; j++){
                vals[j] = Integer.parseInt(arr[j]);
            }
            validationSet.add(vals);
        }
        fileScanner.close();
    }

}
