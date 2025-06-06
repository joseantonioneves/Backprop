using System;
using System.IO;
using RetroPropagacao;

namespace CodingBackProp
{
    class BackPropProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nIniciar demonstração rede neural usanndo retro-propagação ");

            int numInput = 4; // número de características que influenciam o fenômeno
            int numHidden = 5;
            int numOutput = 3; // número de classes para Y (dados que compoem a previsão)
            int numRows = 1000;
            int seed = 1; // parece bom para uma demo :o)

            Console.WriteLine("\nGerando " + numRows +
              " itens de dados artificiais com " + numInput + " características");
            double[][] allData = MakeAllData(numInput, numHidden, numOutput,
              numRows, seed);
            Console.WriteLine("Feito");

            //ShowMatrix(allData, allData.Length, 2, true);

            Console.WriteLine("\nCriando treinamento (80%) e matrizes teste (20%)");
            double[][] trainData;
            double[][] testData;
            SplitTrainTest(allData, 0.80, seed, out trainData, out testData);
            Console.WriteLine("Feito\n");

            Console.WriteLine("Treinando dados:");
            ShowMatrix(trainData, 4, 2, true);
            Console.WriteLine("Testando dados:");
            ShowMatrix(testData, 4, 2, true);

            Console.WriteLine("Criando uma rede neural" + numInput + "-" + numHidden +
              "-" + numOutput);
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;
            Console.WriteLine("\nConfigurando maxEpochs = " + maxEpochs);
            Console.WriteLine("Configurando learnRate = " + learnRate.ToString("F2"));
            Console.WriteLine("Configurando momentum  = " + momentum.ToString("F2"));

            Console.WriteLine("\nIniciando Treinamento");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("Feito");
            Console.WriteLine("\nPesos e vieses finais do modelo de rede neural:\n");
            ShowVector(weights, 2, 10, true);

            //double[] y = nn.ComputeOutputs(new double[] { 1.0, 2.0, 3.0, 4.0 });
            //ShowVector(y, 3, 3, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nPrecisão final nos dados de treinamento = " +
              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("Precisão final nos dados do teste       = " +
              testAcc.ToString("F4"));

            Console.WriteLine("\nTerminando demonstração de retro propagação\n");
            Console.ReadLine();
        } // Main
        public static void ShowMatrix(double[][] matrix, int numRows,
         int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices == true)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }

        public static void ShowVector(double[] vector, int decimals,
          int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        static double[][] MakeAllData(int numInput, int numHidden,
          int numOutput, int numRows, int seed)
        {
            Random rnd = new Random(seed);
            int numWeights = (numInput * numHidden) + numHidden +
              (numHidden * numOutput) + numOutput;
            double[] weights = new double[numWeights]; // pesos e vieses atuais
            for (int i = 0; i < numWeights; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 a 10.0]

            Console.WriteLine("Gerando pesos e vieses:");
            ShowVector(weights, 2, 10, true);

            double[][] result = new double[numRows][]; // Alocando o retorno-resultado
            for (int i = 0; i < numRows; ++i)
                result[i] = new double[numInput + numOutput]; // 1-de-N na última coluna

            NeuralNetwork gnn =
              new NeuralNetwork(numInput, numHidden, numOutput); // gerando NN
            gnn.SetWeights(weights);

            for (int r = 0; r < numRows; ++r) // para cada linha
            {
                // Gerando os inputs aleatórios
                double[] inputs = new double[numInput];

                // loop para geração Randomica dos valores de entrada
                //for (int i = 0; i < numInput; ++i)
                //    inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 a 10.0]
                // ------------------------------------------------------------------------------

                // Lendo dados a partir de um arquivo CSV - 31/12/2019

                // especifica localização e qual arquivo CSV contém os dados
                string dir = System.IO.Path.GetDirectoryName(
                    System.Reflection.Assembly.GetExecutingAssembly().Location); //coleta o diretório de execução da aplicação

                StreamReader stream = new StreamReader(dir + @"\CSVFileInput\teste.csv"); //toma caminho relativo do CSV

                //Leitura das linhas do arquivo CSV
                string linha = null;
                string[] coluna = null;

                while ((linha = stream.ReadLine()) != null)
                {
                    coluna = linha.Split(';');
                    for (int i = 0; i < numInput; i++) inputs[i] = Double.Parse(coluna[i]);
                }
                stream.Close();
                // ---------------------------------------------------------------------------------

                // calculando saídas
                double[] outputs = gnn.ComputeOutputs(inputs);

                // traduzindo saídas para 1-de-N
                double[] oneOfN = new double[numOutput]; // todos 0.0

                int maxIndex = 0;
                double maxValue = outputs[0];
                for (int i = 0; i < numOutput; ++i)
                {
                    if (outputs[i] > maxValue)
                    {
                        maxIndex = i;
                        maxValue = outputs[i];
                    }
                }
                oneOfN[maxIndex] = 1.0;

                // colocando valores de entradas e saídas 1-de-N na linha atual
                int c = 0; // coluna dentro de result[][]
                for (int i = 0; i < numInput; ++i) // inputs
                    result[r][c++] = inputs[i];
                for (int i = 0; i < numOutput; ++i) // outputs
                    result[r][c++] = oneOfN[i];
            } // each row
            return result;
        } // MakeAllData

        static void SplitTrainTest(double[][] allData, double trainPct,
          int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // SplitTrainTest
    } // Program


}  