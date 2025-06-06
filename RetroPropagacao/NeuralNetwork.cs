using System;

namespace RetroPropagacao
{
    public class NeuralNetwork
    {
        private int numInput; // número de nós de entrada
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // entrada-oculto
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights; // oculto-saída
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            this.rnd = new Random(0);
            this.InitializeWeights(); // todos pesos e desvios
        } // ctor

        private static double[][] MakeMatrix(int rows,
            int cols, double v) // helper para ctor, Treinar
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }//MakeMatrix

        //private static double[][] MakeMatrixRandom(int rows,
        //  int cols, int seed) // helper para ctor, treinar
        //{
        //  Random rnd = new Random(seed);
        //  double hi = 0.01;
        //  double lo = -0.01;
        //  double[][] result = new double[rows][];
        //  for (int r = 0; r < result.Length; ++r)
        //    result[r] = new double[cols];
        //  for (int i = 0; i < rows; ++i)
        //    for (int j = 0; j < cols; ++j)
        //      result[i][j] = (hi - lo) * rnd.NextDouble() + lo;
        //  return result;
        //} //MakeMatrixRandom

        private void InitializeWeights() // helper para ctor
        {
            // inicializa pesos e desvios para pequenos valores aleatórios
            int numWeights = (numInput * numHidden) +
                (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }//InitializeWeights

        public void SetWeights(double[] weights)
        {
            // copiar pesos serializados e desvios na matriz weights []
            // para pesos i-h, desvios i-h, pesos h-o, desvios h-o
            int numWeights = (numInput * numHidden) +
                (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Matriz de pesos ruins em SetWeights");

            int k = 0; // pontos dentro do parâmetro weights

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }//SetWeights

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) +
                (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }//GetWeigth

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden]; // nós ocultos somam matriz nula
            double[] oSums = new double[numOutput]; // soma dos nós de saída

            for (int i = 0; i < xValues.Length; ++i) // copia x-values para inputs
                this.inputs[i] = xValues[i];
            // nota: não é necessário copiar "x-values", a menos que você implemente um ToString.
            //       mais eficiente é simplesmente usar o xValues [] diretamente.

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // nota +=

            for (int i = 0; i < numHidden; ++i)  // adicionar vises a somas ocultas
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < numHidden; ++i)   // aplicar ativação
                this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded

            for (int j = 0; j < numOutput; ++j)   // calcular h-o soma dos weight * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // adicionar vieses às somas de saída
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // todas as saídas ao mesmo tempo por eficiência
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput]; //poderia definir um GetOutputs 
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }//ComputeOutputs

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; // aproximação é correta para 30 casas decimais
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }//HyperTan

        /// <summary>
        /// Em matemática, a função softmax, também conhecida como softargmax 
        /// ou função exponencial normalizada, é uma função que recebe como 
        /// entrada um vetor de K números reais e a normaliza em uma distribuição 
        /// de probabilidade que consiste em K probabilidades proporcionais 
        /// às exponenciais dos números de entrada.
        /// </summary>
        /// <param name="oSums">soma das das saídas</param>
        /// <returns></returns>
        private static double[] Softmax(double[] oSums)
        {
            // faz todos os nós de saída de uma só vez para escalar
            // não precisa ser recalculado toda vez

            double sum = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                sum += Math.Exp(oSums[i]);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / sum;

            return result; // agora dimensionado para que xi totalize 1,0
        }//Softmax

        public double[] Train(double[][] trainData, int maxEpochs,
            double learnRate, double momentum)
        {
            // Treinamento usando retro propagação
            // matrizes específicas de retro propagação
            double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // gradientes de peso oculta-para-saída
            double[] obGrads = new double[numOutput];                   // gradientes de desvio de saída

            double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // gradientes de peso entrada-para-oculta
            double[] hbGrads = new double[numHidden];                   // gradientes de desvio da camada oculta

            double[] oSignals = new double[numOutput];                  // sinais locais de saída de gradiente - gradientes sem termos de entrada associados
            double[] hSignals = new double[numHidden];                  // sinais de nó oculto de gradiente local

            // back-prop momentum specific arrays 
            double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
            double[] hPrevBiasesDelta = new double[numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
            double[] oPrevBiasesDelta = new double[numOutput];

            int epoch = 0;
            double[] xValues = new double[numInput]; // entradas
            double[] tValues = new double[numOutput]; // valores alvo
            double derivative = 0.0;
            double errorSignal = 0.0;

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxEpochs / 10; // intervalo para verificação de erro
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 && epoch < maxEpochs)
                {
                    double trainErr = Error(trainData);
                    Console.WriteLine("epoch = " + epoch + "  erro = " +
                        trainErr.ToString("F4"));
                    //Console.ReadLine();
                }

                Shuffle(sequence); // visite cada dado de treinamento em ordem aleatória
                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues); //copiar "xValues" em, saídas de computadas

                    // indices: i = inputs, j = hiddens, k = outputs

                    // 1. calcular sinais do nó de saída (assume softmax)
                    for (int k = 0; k < numOutput; ++k)
                    {
                        errorSignal = tValues[k] - outputs[k];  // Wikipedia uses (o-t)
                        derivative = (1 - outputs[k]) * outputs[k]; // for softmax
                        oSignals[k] = errorSignal * derivative;
                    }

                    // 2. calcular gradientes de peso oculto para saída usando sinais de saída
                    for (int j = 0; j < numHidden; ++j)
                        for (int k = 0; k < numOutput; ++k)
                            hoGrads[j][k] = oSignals[k] * hOutputs[j];

                    // 2b. calcular gradientes de desvio de saída usando sinais de saída
                    for (int k = 0; k < numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

                    // 3. calcular sinais de nós ocultos
                    for (int j = 0; j < numHidden; ++j)
                    {
                        derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // for tanh
                        double sum = 0.0; // necessário somas de sinais de saída vezes pesos ocultos na saída
                        for (int k = 0; k < numOutput; ++k)
                        {
                            sum += oSignals[k] * hoWeights[j][k]; // representa sinal de erro
                        }
                        hSignals[j] = derivative * sum;
                    }

                    // 4. calcular gradientes de peso ocultos de entrada
                    for (int i = 0; i < numInput; ++i)
                        for (int j = 0; j < numHidden; ++j)
                            ihGrads[i][j] = hSignals[j] * inputs[i];

                    // 4b. calcular gradientes de desvio de nó oculto
                    for (int j = 0; j < numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

                    // == atualiza pesos e desvios

                    // atualizar pesos de entrada para ocultos
                    for (int i = 0; i < numInput; ++i)
                    {
                        for (int j = 0; j < numHidden; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
                            ihWeights[i][j] += delta; // seria -= if (o-t)
                            ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta; // salvar para a próxima vez
                        }
                    }

                    // atualizar desvios ocultos
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
                        hBiases[j] += delta;
                        hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // atualizar pesos ocultos-para-saída
                    for (int j = 0; j < numHidden; ++j)
                    {
                        for (int k = 0; k < numOutput; ++k)
                        {
                            double delta = hoGrads[j][k] * learnRate;
                            hoWeights[j][k] += delta;
                            hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // atualizar desvios do nó de saída
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
                        oBiases[k] += delta;
                        oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // cada item de treinamento

            } // while
            double[] bestWts = GetWeights();
            return bestWts;
        } // Train (método)

        private void Shuffle(int[] sequence) // instancia do método
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } // Shuffle

        private double Error(double[][] trainData)
        {
            // erro quadrado médio por item de treinamento
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // primeiros valores "numInput" em trainData
            double[] tValues = new double[numOutput]; // últimos valores "numOutput"

            // percorra cada caso de treinamento. parece como (6,9 3,2 5,7 2,3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // obter valores alvo
                double[] yValues = this.ComputeOutputs(xValues); // saídas usando pesos atuais
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        } // MeanSquaredError

        public double Accuracy(double[][] testData)
        {
            // porcentagem correta usando: vencedor - leva tudo
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // entradas
            double[] tValues = new double[numOutput]; // alvos
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // obter valores x
                Array.Copy(testData[i], numInput, tValues, 0, numOutput); // obter t-values
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // qual célula em yValues tem o maior valor?
                int tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }// Accuracy

        private static int MaxIndex(double[] vector) // helper para Accuracy()
        {
            // índice de maior valor
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }//MaxIndex
    }//NeuralNetwork
}//RetroPropagacao

