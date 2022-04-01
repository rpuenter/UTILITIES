#ifndef  _UTILITIES_H_
#define  _UTILITIES_H_

#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <iomanip>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <assert.h>
#include <random>
#include <functional>
#include <complex>
#include <valarray>
#include <regex>
#include <exception>

#include <thread>
#include <future>
//#include <execution>

#include <signal.h>

#include <spline.h>

#define PI	M_PI	/* pi to machine precision, defined in math.h */
#define TWOPI	(2.0*PI)
#define MYTHROW(a) {std::string error____(a); throw std::runtime_error(error____);}
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#ifdef Success
  #undef Success
#endif
#ifdef tolerance
  #undef tolerance
#endif
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>


// Selection of "useful" functions

class debugUtil
{
private:

   bool isDebug;

public:

    debugUtil(){}
    debugUtil(bool isDebug_){isDebug=isDebug_;}
    ~debugUtil(){}

    void setDebug(){isDebug=true;}
    void unSetDebug(){isDebug=false;}
    void wait();

};

class utilities
{

public:

    // Read a column text file
    static void readFile(const std::string &File_name,
                         std::vector<double> &x,
                         std::vector<double> &u,
                         const std::pair<size_t, size_t> &colsToRead=std::make_pair(0,1),
                         const size_t &thingsToRead=2);


    //Removes white spaces, ",", ";" and tabs from the begining and at the end.
    static void CleanString(std::string &input);

    // Check file extension
    static bool CheckExtension(const std::string &input, const std::string &ext);
    static std::string RemoveExtension(const std::string &input);

    // Splits a string in a specified number of delimited parts
    static void SplitStringN( std::string Input, size_t N,
                             std::vector<std::string> & Out, int & error);

    // This version receives the accepted delimiters as a parameters
    static void SplitStringN(std::string Input,  size_t N,
                             std::vector<std::string> & Out, const std::string &delimiters , int & error);


    // Translates a File stream to a std::stringstream
    static void FileToStream(std::string, std::stringstream &);

    // Checks if there is a comment character at the beggining of a std::string
    static bool isComment(const std::string &line);


    // Translates the simple type within a container
    template<class T1, class T2>
    static std::vector<T1> ContainerCaster(const std::vector<T2> &ToCast)
    {

        std::vector<T1> Casted(0);

        Casted.insert(Casted.begin(),ToCast.begin(),ToCast.end());

        return(Casted);

    }

    template<class T1, class T2>
    static std::vector< std::vector<T1> > ContainerCaster(const std::vector< std::vector<T2> > &ToCast)
    {

        std::vector< std::vector<T1> > Casted(ToCast.size());

        for(size_t i(0); i < ToCast.size(); ++i) Casted[i] = ContainerCaster<T1,T2>(ToCast[i]);

        return(Casted);

    }

    template<class T1, class T2>
    static std::vector< std::vector< std::vector<T1> > > ContainerCaster(const std::vector< std::vector< std::vector<T2> > > &ToCast)
    {

        std::vector< std::vector< std::vector<T1> > > Casted(ToCast.size());

        for(size_t i(0); i < ToCast.size(); ++i) Casted[i] = ContainerCaster<T1,T2>(ToCast[i]);

        return(Casted);

    }


    // Translates a number to a string
    template<class T>
    static std::string NumToString(const T &num,const size_t &setPrec=10)
    {

        std::stringstream stream;
        stream << std::setprecision(setPrec) << num;
        std::string str( stream.str() );

        return(str);

    }


    // Translates a number to a string choosing precision
    template<class T>
    static std::string NumToStringP(const T &num, int Prec)
    {

        std::stringstream stream;
        stream << std::setprecision(Prec) << num;
        std::string str( stream.str() );

        return(str);

    }


    template<class T>
    static std::vector<std::string> NumToString(const std::vector<T> &num)
    {

        std::vector<std::string> str(num.size(),"");
        for (size_t i(0); i < num.size(); ++i) str[i] = NumToString(num[i]);

        return(str);

    }


    // Translates std::string to number
    static int stoi_w(const std::string& str, int* p_value, std::size_t* pos = 0, int base = 10)
    {
        // wrapping std::stoi because it may throw an exception

        try {
            *p_value = std::stoi(str, pos, base);
            return 0;
        }

        catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
            return -1;
        }

        catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << std::endl;
            return -2;
        }

        catch (const std::exception& e)
        {
            std::cerr << "Undefined error: " << e.what() << std::endl;
            return -3;
        }
    }

    static int stod_w(const std::string& str, double* p_value, std::size_t* pos = 0)
    {
        // wrapping std::stod because it may throw an exception

        try {
            *p_value = std::stod(str, pos);
            return 0;
        }

        catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
            return -1;
        }

        catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << std::endl;
            return -2;
        }

        catch (const std::exception& e)
        {
            std::cerr << "Undefined error: " << e.what() << std::endl;
            return -3;
        }
    }

    static int stof_w(const std::string& str, float* p_value, std::size_t* pos = 0)
    {
        // wrapping std::stod because it may throw an exception

        try {
            *p_value = std::stof(str, pos);
            return 0;
        }

        catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
            return -1;
        }

        catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << std::endl;
            return -2;
        }

        catch (const std::exception& e)
        {
            std::cerr << "Undefined error: " << e.what() << std::endl;
            return -3;
        }
    }


    static int strToInt(const std::string &str)
    {

        int value = 0.;
        int errVal = stoi_w(str,&value);
        if (errVal != 0)
        {
            std::cout << "Error in utilities::strToInt()\n";
        }
        return(value);

    }

    static float strToFloat(const std::string &str)
    {

        float value = 0.;
        int errVal = stof_w(str,&value);
        if (errVal != 0)
        {
            std::cout << "Error in utilities::strToFloat()\n";
        }
        return(value);

    }

    static double strToDouble(const std::string &str)
    {

        double value = 0.;
        int errVal = stod_w(str,&value);
        if (errVal != 0)
        {
            std::cout << "Error in utilities::strToDouble()\n";
        }
        return(value);

    }




    // Matrix to vector and viceversa reshaping
    // Separate in two different functions for ease of use?
    template<class T>
    static void Reshape(std::vector<T> &vec,
                        std::vector< std::vector<T> > &matrix,
                        const short int &mode)
    {

        size_t M( matrix.size() );
        std::vector<size_t> N(M);

        switch (mode)
        {
        case 0:
        {

            /*

        Stores an input std::vector<std::vector<>> as a concatenated
        std::vector<>.

       */

            vec.resize(0);

            for (size_t i(0); i < M; ++i)
            {

                //int N = matrix[i].size();
                vec.insert(vec.end(), matrix[i].begin(), matrix[i].end());
            }

            break;
        }
        case 1:
        {

            /*

         Stores an input std::vector<> as a std::vector<std::vector<>>  whose
         size must be specified by outside allocation.

         N is the inner dimension. M the outer

       */


            for (size_t i(0); i < M; ++i)
            {

                N[i] = matrix[i].size();
                /*cout << "Matrix size [i] = " << N[i] << endl;*/
                std::vector<T> extracted(N[i]);

                for (size_t j(0); j < N[i]; ++j)
                {
                    int NN = 0;
                    for (int jj = 0; jj < i; ++jj) { NN = NN + N[jj]; }
                    int k = NN + j;

                    extracted[j] = vec[k];

                }

                matrix[i]= extracted;
            }

            break;

        }
        default:

            MYTHROW("Mode not recognized!\n");
            break;

        }

    }




    // Sort a vector from lowest to highest and give the indexing


    // Previous auxiliary definitions
    template < typename P1, typename P2 >
    class less_first
    {
        typedef std::pair< P1, P2 > value_type;
    public:
        bool operator() ( const value_type& lhs , const value_type & rhs )
        {
            return lhs.first < rhs.first;
        }
    };


    template<class T>
    static void sort_index(const std::vector<T> &to_sort, std::vector<T> &sorted,
                           std::vector<size_t> &idx)
    {

        /*

            Sorts a std::vector<float> from lowest to highest value
            Provides the reordering vector such that:
                    - idx[i] = rank of to_sort[i]

      */

        size_t size(to_sort.size());

        std::vector< std::pair<T,int> > data(0);

        data.reserve(size);

        for(size_t i(0); i < size; ++i)
        {
            data.push_back(std::make_pair(to_sort[i],i));
        }

        std::sort(data.begin(), data.end(),less_first<double,int>());

        idx.clear();     idx.reserve(size);

        sorted.clear();  sorted.reserve(size);


        for(size_t i(0); i < size; ++i)
        {
            idx.push_back(data[i].second);
            sorted.push_back(data[i].first);
        }




    }

    // Sort a vector and reorder several associated ones accordingly
    template<class T>
    static std::vector< std::vector<T> > SortMatrix(size_t Idx, const std::vector< std::vector<T> > &matrix)
    {
        /*
      Reorders every vector of a matrix according to the result of sort(matrix[Idx])
    */

        std::vector<size_t> indexing(matrix[Idx].size());
        std::vector< std::vector<T> > newMatrix(matrix.size());

        size_t N(matrix[Idx].size());

        sort_index<T>(matrix[Idx],newMatrix[Idx],indexing);

        for(size_t i(0); i < matrix.size(); ++i)
        {
            if(i != Idx)
            {
                newMatrix[i].resize(N);

                if(matrix[i].size() != N)
                {
                    MYTHROW("Vector sizes not consistent in utilites::SortMatrix\n");
                    break;
                }

                for(size_t j(0); j < N; ++j)
                {
                    newMatrix[i][j] = matrix[i][indexing[j]];
                }

            }

        }

        return(newMatrix);

    }

    // Sort a vector and reorder an associated one
    template<class T1, class T2>
    static std::vector<T2> Sort2Vectors(const std::vector<T1> &order,const std::vector<T2> &ToOrder)
    {
        /*
        Reorders vector ToOrder according to the result of sort(order)

        T1 must be a simple type that can actually be sorted, while T2
        can be any type of associated data
     */

        if(ToOrder.size() != order.size())
        {
            MYTHROW("Vector sizes not consistent in utilites::Sort2Vectors\n");
        }

        size_t size(order.size());
        std::vector<size_t> indexing(size,0);
        std::vector<T1> orderSorted(size,0.);
        std::vector<T2> NewToOrder(size,0.);

        sort_index<T1>(order,orderSorted,indexing);

        for(size_t i(0); i < size; ++i)
        {
            NewToOrder[i] = ToOrder[indexing[i]];
        }

        return(NewToOrder);

    }
    
    // Sort a vector and reorder an associated one
    template<class T1, class T2>
    static void Sort2Vectors(std::vector<T1> &order,std::vector<T2> &ToOrder)
    {
        /*
        Reorders vector ToOrder according to the result of sort(order)

        T1 must be a simple type that can actually be sorted, while T2
        can be any type of associated data
     */

        if(ToOrder.size() != order.size())
        {
            MYTHROW("Vector sizes not consistent in utilites::Sort2Vectors\n");
        }

        std::vector<size_t> indexing(order.size(),0);
        std::vector<T1> orderSorted(order.size());
        std::vector<T2> NewToOrder(order.size());

        sort_index<T1>(order,orderSorted,indexing);

        for(size_t i(0); i < order.size(); ++i)
        {
            NewToOrder[i] = ToOrder[indexing[i]];
        }

        order   = orderSorted;
        ToOrder = NewToOrder;

    }
    // Translates a vector to an array and viceversa
    template<class T1, class T2>
    static T1 *VectorToArray(const std::vector<T2> &dataVec)

    {

        size_t dataVecSize( dataVec.size() );

        T1 *dataArray = new T1[dataVecSize];

        memcpy( dataArray, &dataVec[0], sizeof( T1 ) * dataVecSize );

        return(dataArray);


    }


    template<class T1, class T2>
    static std::vector<T2> ArrayToVector(T1 *dataArray,size_t ArraySize)

    {

        std::vector<T2> dataVec(dataArray ,dataArray  + ArraySize);

        return(dataVec);

    }

    // Elementwise vector addition
    template<class T1, class T2, class T3>
    static std::vector<T3> VecAdd(const std::vector<T1> &v1, const std::vector<T2> &v2)

    {

        size_t n1(v1.size()),n2(v2.size());
        T3 kk(0.);


        if (n1 != n2) MYTHROW("Vector dimensions not consistent!\n");

        std::vector<T3> v3(0);

        for (size_t i(0); i < n1; ++i)
        {
            kk = v1[i] + v2[i];
            v3.push_back(kk);
        }

        return(v3);

    }

    // Scale a vector with another number
    template<class T1, class T2, class T3>
    static std::vector<T1> VecMul(const std::vector<T2> &v1, const T3 &scale)

    {

        size_t n1( v1.size());

        std::vector<T2> v3(v1);
        for (size_t i(0); i < n1; ++i)
        {
            v3[i] *= scale;
        }

        return(v3);

    }

    // Scalar product of vectors
    template<class T1, class T2, class T3>
    static T1 ScalarVecMul(const std::vector<T2> &v1, const std::vector<T3> &v2)

    {

        size_t n1(v1.size()),n2(v2.size());


        if (n1 != n2) MYTHROW("Vector dimensions not consistent!\n");

        T1 scalar(0.0);
        for (size_t i(0); i < n1; ++i)
        {
            scalar += v1[i]*v2[i];
        }

        return(scalar);

    }

    // Scalar product of vectors with an inner product defined as a diagonal form
    template<class T1,class T2,class T3,class T4>
    static T1 ScalarVecMul(const std::vector<T2> &v1, const std::vector<T3> &v2,const std::vector<T4> &innerProd)

    {

        size_t n1(v1.size()),
                n2(v2.size());


        if (n1 != n2) MYTHROW("Vector dimensions not consistent!\n");

        T1 scalar(0.0);
        for (int i(0); i < n1; ++i)
        {
            scalar += v1[i]*v2[i]*innerProd[i];
        }


        return(scalar);

    }

    // Scalar product of arrays
    template<class T1, class T2, class T3>
    static T1 ArrMul(T2 *v1, T3 *v2, int SIZE)

    {

        T1 scalar(0.0);
        for (size_t i(0); i < SIZE; ++i)
        {
            scalar += v1[i]*v2[i];
        }

        return(scalar);
    }

    // Elementwise product of vectors
    template<class T1,class T2,class T3>
    static std::vector<T1> VecMul(const std::vector<T2> &v1, const std::vector<T3> &v2)

    {

        size_t n1(v1.size()),
            n2(v2.size());


        if (n1 != n2) MYTHROW("Vector dimensions not consistent!\n");

        std::vector<T1> v3(n1,0);

        for (size_t i(0); i < n1; ++i)
        {
            v3[i] = v1[i]*v2[i];
        }

        return(v3);
    }

    // Diadic product of vectors
    template<class T1, class T2, class T3>
    static std::vector< std::vector<T1> > DiadicVecMul(const std::vector<T2> &v1, const std::vector<T3> &v2)

    {

        size_t n1(v1.size()),
               n2(v2.size());


        if (n1 != n2) MYTHROW("Vector dimensions not consistent!\n");

        std::vector< std::vector<T1> > matrix(n1);


        for (size_t i(0); i < n1; ++i)
        {

            matrix[i].resize(n1);

            for (int j(0); j < n1; ++j )
            {
                matrix[i][j] = v1[i]*v2[j];
            }

        }

        return(matrix);

    }


    // Standard rectangular matrix product
    template<class T1, class T2, class T3>
    static std::vector< std::vector<T1> > MatMul(const std::vector< std::vector<T2> > &m1,
                                       const std::vector< std::vector<T3> > &m2)

    {

        size_t L(m1.size()),
                M(m1[0].size()),
                N(m2[0].size());


        if (m2.size() != L) MYTHROW("Matrix rows not consistent!\n");
        if (M != L) MYTHROW("Matrix columns not consistent!\n");
        if (M != N) MYTHROW("Matrix dimensions not consistent!\n");

        std::vector< std::vector<T1> > m3(L);

        //Transpose m2
        std::vector< std::vector<T3> > m2T = m2;
        for(size_t i(0); i < L; ++i)
        {
            for(size_t j(0); j < N; ++j )
            {
                m2T[i][j] = m2[j][i];
            }
        }

        for(size_t i(0); i < L; ++i)
        {
            m3[i].resize(M,0.);

            for(size_t j(0); j < N; ++j )
            {
                m3[i][j] = ScalarVecMul<T1,T2,T3>(m1[i],m2T[j]);
            }
        }


        return(m3);
    }


    // Left matrix by vector product
    template<class T1, class T2, class T3>
    static std::vector<T1>  MatMul(const std::vector< std::vector<T2> > &m, const std::vector<T3>  &v)

    {

        size_t L(m.size()),
                M(m[0].size()),
                N(v.size());

        if (N != M) MYTHROW("Matrix and vector dimensions not consistent!\n");

        std::vector<T1> v2(N,0);

        for (size_t i(0); i < L; ++i)
        {
            v2[i] = VecMul<T1,T2,T3>(m[i],v);
        }

        return(v2);

    }


    // Function quadrature
    template<class T1,class T2>
    static T1 FuncQuad(const std::vector<T1> &fun, const std::vector<T2> &base)
    {

        /*
      Integrates v1 over the basis defined by v2 with the trapezoidal rule
    */

        size_t n1(fun.size()),
               n2(base.size());


        if(n1 != n2) MYTHROW("Vector dimensions not consistent in utilities::FuncQuad()!\n");

        if(n1 == 0) MYTHROW("Empty input in utilities::FuncQuad()!\n");

        T1 integral(0.0);

        if(n1==1)
        {
            // Do nothing
        }
        else if(n1==2)
        {
            T1 dx = base[1] - base[0];
            T2 s  = fun[1] + fun[0];

            integral = 0.5*s*static_cast<T1>(dx);
        }
        else
        {
            for (size_t i(0); i < (n1-1); ++i)
            {
                T1 dx = base[i+1] - base[i];
                T2 s  = fun[i+1] + fun[i];
                integral += 0.5*s*static_cast<T1>(dx);
            }
        }

        return(integral);

    }

    template<class T>
    static T FuncQuad(std::function<T(T)> fun, const std::vector<T> &base)
    {
        std::vector<T> funV(0);
        for (size_t i(0); i < base.size(); ++i)
        {
            funV.push_back(fun(base[i]));
        }

        return FuncQuad<T,T>(funV,base);
    }

    // Function quadrature using the Monte Carlo method with a given pont distribution
    // Does not require a sorted array
    template<class T>
    static T QuasiMonteCarloQuadrature(const std::vector<T> &fun, const std::vector<T> &base)

    {

        /*
      Integrates v1 over the basis defined by v2 with the Monte Carlo method
    */

        size_t n1(fun.size()),
               n2(base.size());


        if(n1 != n2) MYTHROW("Vector dimensions not consistent in utilities::FuncQuad()!\n");

        if(n1 == 0) MYTHROW("Empty input in utilities::QuasiMonteCarloQuadrature()!\n");

        double max = *std::max_element(base.begin(),base.end()),
               min = *std::min_element(base.begin(),base.end()),
               vol = max-min,
               factor = vol/static_cast<T>(n1);

        T integral = std::accumulate(fun.begin(), fun.end(), 0.)*factor;

        return(integral);

    }

    // Function quadrature using the Monte Carlo method in 1D
    template<class T>
    static T MonteCarloQuadrature(std::function<T(T)> fun,const T &low,const T &up,size_t nsamples)

    {

        std::vector<T> rd = randomArray(nsamples,up,low);

        std::vector<T> fval(0);
        for (size_t i(0); i < nsamples; ++i)
        {
            fval.push_back(fun(rd[i]));
        }


         return QuasiMonteCarloQuadrature(rd,fval);

    }


    // Generate integral function F=int_x0^x(fdx)
    template<class T1,class T2>
    static std::vector<T1> IntegralFunction(const std::vector<T1> &fun, const std::vector<T2> &base,const T1 &inival=0.)

    {
        std::vector<T1> integral(1,inival);
        std::vector<T1> localFun(1,fun[0]);
        std::vector<T2> localBase(1,base[0]);

        for(size_t idx(1); idx<fun.size();++idx)
        {
            localFun.push_back(fun[idx]);
            localBase.push_back(base[idx]);
            integral.push_back(inival+FuncQuad<T1,T2>(localFun,localBase));
        }
        return integral;
    }

    template<class T1,class T2>
    static std::vector<T1> IntegralFunction(std::function<T1(const T1 &)> fun, const std::vector<T2> &base,const T1 &inival=0.)
    {

        std::vector<T1> funV(0);
        for(size_t idx(0); idx<base.size();++idx)
        {
            funV.push_back(fun(base[idx]));
        }

        return(IntegralFunction<T1,T2>(funV,base));
    }




    // Size of an array. DOES NOT WORK WITH DINAMICALLY ALLOCATED ARRAYS
    template<class T>
    static size_t ArraySize(T* array)
    {
        size_t size = sizeof(array)/sizeof(T);
        return(size);
    }


    /*
  // Matricial product
  template <class T>
  static std::vector< std::vector<T> > MatMul(std::vector< std::vector<T> > m1, std::vector< std::vector<T> > m2)

  {


  // m1(L,M)
  // m2(M,N)
  // m3(L,N)


  int L,M,N,kk,Lcount,Mcount,Ncount;

  L=m1.size();
  M=m1[0].size();
  N=m2[0].size();

  kk=m2.size();

  if(kk != M) MYTHROW("Matrices are not compatible!");

  std::vector<T> m3v(N,0);
  std::vector< std::vector<T> > m3(m3v);



  return m3;



  }

  */

    // Auxiliary loop function used recursively in CartesianProduct()
    template<class T>
    static void CartesianLoop(size_t N,const std::vector< std::vector<T> > &InContainer,std::vector< std::vector<T> > &OutContainer,std::vector<T> &AuxVector,int depth)
    {

        ++depth;

        for(size_t i(0); i < InContainer[depth].size(); ++i)
        {
            AuxVector[depth] = InContainer[depth][i];


            if(depth == (N-1))
            {
                OutContainer.push_back(AuxVector);
            }
            else
            {
                CartesianLoop(N,InContainer,OutContainer,AuxVector,depth);
            }

        }


    }

    // Cartesian product of the vectors in container. Useful to generate mesh points
    template<class T>
    static std::vector< std::vector<T> > CartesianProduct(const std::vector< std::vector<T> >  &InContainer)
    {

        size_t N(InContainer.size());               // Number of dimensions

        std::vector< std::vector<T> > OutContainer(0);     // Output value
        std::vector<T> AuxVector(N);                  // Auxiliary vector

        int depth(-1);                           // Nested CartesianLoop level

        CartesianLoop(N,InContainer,OutContainer,AuxVector,depth);

        return(OutContainer);


    }


    // Cartesian components of a hypersphere of dimension "N", given radius
    // over a non-strictly uniform mesh of "P" on the "N-1" stereographic angles
    template<class T>
    static void HyperSphere(int N, int P, T r, std::vector< std::vector<T> > &x)
    {

        /*

             Evaluates the cartesian coordinates "x" of a N-hypershere defined by:
                     - r: radius
                     - PHI[i]: Stereographic angles. PHI.size() = N-1
                     - P: Number of mesh points per angle (could be a vector
                         in an overloaded version)
                     - P: Number of points


    NOTE: Coordinate shift between n-spherical coordinates and cartesian

    x0 = r*std::cos(phi0)
    x1=r*std::sin(phi0)*std::cos(phi1)
    x2=r*std::sin(phi0)*std::sin(phi1)*std::cos(phi2)
    :
  x(i)=r*PI(K)(std::sin(phiK))*std::cos(phii); K=0,i-1
  :
  x(N-2)=r*PI(K)(std::sin(phi(K)))*std::cos(phi(N-3)); K=0,N-3
  x(N-1)=r*PI(K)(std::sin(phiK)); K=0,N-2

  */


        /*
        M is the number of mesh points per angle so that P = (N-1)*M
        Then: M = P/(N-1). as this result will not be an exact integer in general,
        M will be taken as the integer part of this value. The remainding points will be taken randomly
     */


        size_t M(0),remain(0),Pred(0);

        // Limit the scope of the div_t structure
        {
            div_t divresult;
            divresult = div(P,N-1);
            M = divresult.quot;
            remain = divresult.rem;
        }
        Pred = (N-1)*M;

        // Possible values for the sine of a single spherical angle between (0,pi/2)
        std::vector<T> sin_phi(M,0);
        std::vector< std::vector<T> >  SinPhi(Pred);

        double phi(0.0);
        if(M > 1)
        {
            for (size_t i(0); i < M; ++i)
            {
                phi = i*0.5*std::acos(-1)/(M-1);
                sin_phi[i] =std::sin(phi);
            }



            // Build the uniformly spaced mesh: Pred points of N-1 coordinates that can take M values each
            // Cartesian product of (N-1) elements that can take M values


            std::vector< std::vector<T> > AnglesContainer(Pred);
            for(size_t i(0); i < Pred; ++i)
            {
                AnglesContainer[i] = sin_phi;
            }

            SinPhi = CartesianProduct(AnglesContainer);

        }
        else
        {

            // If only one angle is asked for, give the central one
            sin_phi[0] = 0.25*std::acos(-1);

            for(size_t i(0); i < Pred; ++i)
            {
                SinPhi[i] = sin_phi;
            }


        }

        // Compute the cartesian coordinates

        // In SinPhi[i][j]
        // "i" is the coordinate index counter
        // "j" is the node counter

        // Clean useless containers
        sin_phi.clear();
        x.clear();


        x.resize(Pred);

        for (size_t point(0); point < Pred; ++point)
        {

            x[point].resize(N);


            x[point][0] = r*(T)(std::sqrt(1-SinPhi[point][0]*SinPhi[point][0]));


            for (size_t dim(1); dim < (N-1); ++dim)
            {

                x[point][dim] = (T)(1.0);
                for (size_t l(0); l < (dim-1); ++l)
                {
                    x[point][dim] = x[point][dim]*SinPhi[point][l];
                }

                if (dim == (N-2))
                {
                    x[point][dim  ] = r*x[point][dim]*std::sqrt(1-SinPhi[point][dim]*SinPhi[point][dim]);
                    x[point][dim+1] = r*x[point][dim]*SinPhi[point][dim  ];
                }
                else
                {
                    x[point][dim] = r*x[point][dim]*std::sqrt(1-SinPhi[point][dim]*SinPhi[point][dim]);
                }

            }



        }


        // Check if there are any points that having different spherical coordinates have the same cartesian coordinates,
        // if so, take them out  to be substituted by randomly generated ones
        x = RemoveRepeatedElements(x);
        remain =  remain + ( x.size() - Pred );

        // Generate the remaining points
        for(size_t point(0); point < remain; ++point)
        {
            std::vector<T> Aux(N);
            T Aux2(0);

            for(size_t dim(0); dim < N-1; ++dim)
            {

                Aux[dim] = BoundedGaussian((T)(0.1*r*r),(T)(0.9*r*r),(T)0,(T)1);

                Aux2 = Aux2 + Aux[dim]*Aux[dim];

            }

            Aux[N] = std::sqrt(r*r - Aux2 );

            x.push_back(Aux);

        }



    }

    // Check for repeated elements in a container and remove them. The output is sorted.
    template<class T>
    static std::vector<T> RemoveRepeatedElements(const std::vector<T> &container)
    {

        std::vector<T> newContainer(0);
        newContainer = container;

        sort(newContainer.begin(),newContainer.end());
        newContainer.erase( unique(newContainer.begin(),newContainer.end() ), newContainer.end() );


        return(newContainer);
    }


    // Generate a random variable with an interval [a,b] bounded normal distribution
    template<class T>
    static T BoundedGaussian(T a,T b,T mean, T sigma)
    {

        srand ( time(NULL) );

        T x = rand();

        T out;

        out = a + ( b - a )*exp(-0.5*(x-mean)*(x-mean)/(sigma*sigma));

        return(out);

    }


    // Generate a random numbers array
    template<class T>
    static std::vector<T> randomArray(const size_t &siz, const T &upperlim,const T &lowerlim)
    {

        std::random_device rd;  // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator

        std::vector<T> out(0);

        if(typeid(T)==typeid(int))
        {
            std::uniform_int_distribution<int> distr(lowerlim,upperlim);
            for(size_t idx(0); idx < siz; ++idx) out.push_back(distr(eng));
        }
        else
        {
            std::uniform_real_distribution<T> distr(lowerlim,upperlim);
            for(size_t idx(0); idx < siz; ++idx) out.push_back(distr(eng));
        }

        return(out);

    }

    // Add noise to a signal
    template<class T>
    static std::vector<T> AddNoise(const std::vector<T> &signal, const T &SignalToNoise)
    {
        std::vector<T> ones(signal.size(),1.);
        T scale = 0.5*std::sqrt(SignalToNoise);
        std::vector<T> noise =  randomArray(signal.size(),scale,-scale);
        return(VecMul<T,T,T>(signal,VecAdd<T,T,T>(ones,noise)));
    }

    // Generate a sorted random numbers array
    template<class T>
    static std::vector<T> randomArraySorted(size_t siz, const T &upperlim,const T &lowerlim)
    {
        std::vector<T> out = randomArray(siz,upperlim,lowerlim);
        sort(out.begin(),out.end());
        return(out);
    }

    // Auxiliary fuction for generating point distributions
    template<class T>
    static std::vector<T> aux4Dist(const std::vector<T> &distribution, const T &upperlim,const T &lowerlim)
    {

        std::vector<T> out(0);

        T range(upperlim-lowerlim);

        // Normalize the spacing
        std::vector<T> normalizedDistribution =  NormalizeSet(distribution);

        // Add the spacing to the output vector
        for(size_t idx(0); idx < distribution.size(); ++idx)
        {
            out.push_back(lowerlim+range*normalizedDistribution[idx]);
        }

        // Check that the upper lim is achieved
       // assert(std::abs(out.back() - upperlim)<1.e3*std::numeric_limits<T>::epsilon());


        return(out);
    }

    // Generate an evenly distribution of points
    template<class T>
    static std::vector<T> equispacedPointDistribution(size_t siz, const T &upperlim,const T &lowerlim)
    {
        std::vector<T> out(siz);

        T range(upperlim-lowerlim),
          dx(range/(siz-1));

        for(size_t idx(0); idx < siz; ++idx)
        {
           out[idx] = lowerlim+idx*dx;
        }

        // Check that the upper lim is achieved
      //  assert(std::abs(out.back() - upperlim)<1.e3*std::numeric_limits<T>::epsilon());



        return(out);

    }

    // Generate a harmonic distribution of points. Phase is in degrees
    template<class T>
    static std::vector<T> harmonicPointDistribution(size_t siz, const T &upperlim,const T &lowerlim,const T &amplitude,const int &nHarms,const T &phase)
    {
        std::vector<T> distribution(siz,0.);

        T phaserad = 180.*phase/PI,
          iSiz     = 1./T(siz-1),
          dx0      = std::min(iSiz,amplitude); // Arbitrary minimum dx. Gonna be squashed in aux4dist anyway

        for(size_t idx(0); idx < siz-1; ++idx)
        {
            // Square the function to avoid negative values, and halve the period            
            T dx = amplitude*std::sin(PI*nHarms*T(idx)*iSiz-0.5*phaserad)*std::sin(PI*nHarms*T(idx)*iSiz-0.5*phaserad)+dx0;
            distribution[idx+1]=distribution[idx]+dx;
        }

        return(aux4Dist(distribution,upperlim,lowerlim));

    }

    // Generate an exponentially growing distribution of points
    template<class T>
    static std::vector<T> exponentialPointDistribution(size_t siz, const T &upperlim,const T &lowerlim,const T &amplitude,const T &growth)
    {
        std::vector<T> distribution(siz,0.);

        T iSiz = 1./T(siz-1),
          dx0  = std::min(iSiz,amplitude); // Arbitrary minimum dx. Gonna be squashed in aux4dist anyway

        for(size_t idx(0); idx < siz-1; ++idx)
        {
            T dx = amplitude*exp(growth*idx)+dx0;
            distribution[idx+1]=distribution[idx]+dx;
        }

        return(aux4Dist(distribution,upperlim,lowerlim));
    }

    template<class T>
    static std::vector<T> TriDiagSys(const std::vector<T> &d, // Source term
                                const std::vector<T> &a, // Lower diagonal
                                const std::vector<T> &b, // Main diagonal
                                const std::vector<T> &c) // Upper diagonal
    {

        /*
    Solves a tridiagonal linear system of equations using Thomas' algorithm.
      */

        // Check dimensions
        size_t N ( b.size() );

        if( (N!=d.size()) || ((N-1)!=a.size()) || ((N-1)!=c.size()))
            MYTHROW("Wrong matrix dimensions\n");

        // Create the temporary vectors
        // Note that this is inefficient as it is possible to call
        // this function many times. A better implementation would
        // pass these temporary matrices by non-const reference to
        // save excess allocation and deallocation
        std::vector<T> c_star(N, 0.0),
                d_star(N, 0.0),
                f     (N,0.0);

        // This updates the coefficients in the first row
        // Note that we should be checking for division by zero here
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];

        // Create the c_star and d_star coefficients in the forward sweep
        for (size_t i=1; i<N; i++)
        {
            T m = 1.0 / (b[i] - a[i] * c_star[i-1]);
            c_star[i] = c[i] * m;
            d_star[i] = (d[i] - a[i] * d_star[i-1]) * m;
        }

        // This is the reverse sweep, used to update the solution vector f
        for (size_t i=N-1; i-- > 0; )
        {
            f[i] = d_star[i] - c_star[i] * d[i+1];
        }

        return(f);


    }


    template<class T>
    static std::vector< std::vector<T> > SplineCoefs(const std::vector<T> &x, const std::vector<T> &f)
    {

        /*
    Gives the coefficients of the natural cubic spline interpolant defined by f(x)
    (Goddammned c style notation!!!)

    "Wolfram Mathworld"
      */

        size_t N(x.size()); // Total number of elements in x


        std::vector<T>   DD(N  ,(T) 0.);  //

        std::vector<T>    C(N-1,(T) 1);
        std::vector<T>    A(N-1,(T) 1);

        std::vector<T>    D(N  ,(T) 4);
        D[0]   = (T)2.;
        D[N-1] = (T)2.;


        std::vector<T>    B(N  ,(T) 0);
        B[0] = 3.*(f[1]-f[0]);
        for(size_t i(1); i < N-1; ++i)
        {
            B[i] = 3.*(f[i+1]-f[i-1]);
        }
        B[N-1] = 3.*(f[N-1]-f[N-2]);


        // Solve

        DD = TriDiagSys(B,A,D,A);

        std::vector< std::vector<T> > Coef(4);
        Coef[0].resize(N-1);
        Coef[1].resize(N-1);
        Coef[2].resize(N-1);
        Coef[3].resize(N-1);

        for(size_t i(0); i < N-1; ++i)
        {
            Coef[0][i] = f[i];
            Coef[1][i] = DD[i];
            Coef[2][i] = 3.*(f[i+1] - f[i]) - 2.*(DD[i] + DD[i+1]);
            Coef[3][i] = 2.*(f[i] - f[i+1]) + DD[i] + DD[i+1];
        }

        return(Coef);

    }



    template<class T>
    static T SplineEval(const std::vector< std::vector<T> > &Coef, T dx, T dx1, int coefIdx)

    {

        /*
    Evaluates the cubic spline interpolant on xNew using the
    previously computed coefficients

    F(x) = ai + bi t +ci t2 +di t3

        t =  (x - x[i])/(x[i+1]-x[i]) = dx/dx1

        ai = Coefs0, bi = Coefs1, ci = Coefs2, di = Coefs3

    "Wolfram Mathworld"
      */


        T t( dx/dx1 );

        T F ( Coef[0][coefIdx] + Coef[1][coefIdx]*t + Coef[2][coefIdx]*t*t + Coef[3][coefIdx]*t*t*t );

        return(F);

    }
    
    template<class T>
    static std::vector<T>  QuadCoefs(const std::vector<T> &x, const std::vector<T> &f)
    {
        /*
         Evaluates the coefficients of a quadratic over three points (x[0],f[0]), (x[1],f[1]), and (x[2],f[2]).
         
         f(x) = Coef[0] + Coef[1] x + Coef[2] x^2
         
         Calculation:
         
         f(x) = c + b(x-x[1]) + a (x-x[1])^2
         
         c = f[1]
         
         (a ,b)= M^(-1) ((f[2] -f[1] ) , (f[0] -f[1])); where M is the matrix resulting from substituting the equation
                                                        in x[0] and x[2].


       */

        std::vector<T> Coeffs(3,0.0),	// Coefficients vector
                m(4,0.0);		// Inverse matrix of the system of equations

        T a(0.0),
                b(0.0),
                c(f[1]),
                det(0.0);

        det = (x[0]-x[1])*(x[2]-x[1])*(x[2]-x[0]);

        // Hard coded elements of the adjuvant matrix.

        m[0] = x[0]-x[1];
        m[1] = x[1]-x[2];
        m[2] = -m[0]*m[0];
        m[3] = m[1]*m[1];

        a = m[0]*(f[2]-f[1]) + m[1]*(f[0]-f[1]);
        b = m[2]*(f[2]-f[1]) + m[3]*(f[0]-f[1]);

        a /= det;
        b /= det;

        Coeffs[0] = c - b*x[1] + a*x[1]*x[1];
        Coeffs[1] = b - 2.*a*x[1];
        Coeffs[2] = a;


        return(Coeffs);
    }
    
    template<class T>
    static std::vector<T>  QuadTCoefs(const std::vector<T> &x, const std::vector<T> &f,T tangent,int pFlag)
    {
        /*
         Evaluates the coefficients of a quadratic over two points (x[0],f[0]), and (x[1],f[1]), and
         specifying a tangent value at the first point (pFlag = 0), or at the last one(pFlag =1).
         
         f(x) = Coef[0] + Coef[1] x + Coef[2] x^2
         
         Calculation:
         
           pFlag = 0:

             f(x)  = c + b(x-x[0]) + a (x-x[0])^2
             f'(x) = b + 2 a (x-x[0])
             
             c = f[0]

             b = tangent

             a = [ f[1]-f[0] - b (x[1]-x[0]) ]/(x[1]-x[0])^2

           pFlag = 1:

             f(x)  = c + b(x-x[1]) + a (x-x[1])^2
             f'(x) = b + 2 a (x-x[1])

             c = f[1]

             b = tangent
             
             a = [ f[0]-f[1]-b(x[0]-x[1]) ]/(x[1]-x[0])^2

       */



        std::vector<T> Coeffs(3,0.0); // Coefficients vector

        T a(0.0),
          b(tangent),
          c(0.0);

        int index1(0),index2(1);

        if(pFlag == 1)
        {
            ++index1;
            --index2;
        }

        if( (pFlag < 0) || (pFlag > 1) )
            MYTHROW("Problems in utilities::QuadTCoefs(): Wrong pFlag Value!\n");

        c = f[index1];

        a = ( f[index2]-f[index1] - b*(x[index2]-x[index1]))*pow((x[1]-x[0]),-2.);

        Coeffs[0] = c - b*x[index1] + a*x[index1]*x[index1];
        Coeffs[1] = b - 2.*a*x[index1];
        Coeffs[2] = a;


        return(Coeffs);


    }
    
    template<class T>
    static std::vector<T>  CubicTCoefs(const std::vector<T> &x, const std::vector<T> &f,T tangent,int pFlag)
    {
        /*
         Evaluates the coefficients of a cubic over three points (x[0],f[0]),(x[1],f[1]), and (x[2],f[2]),
         specifying a tangent value at the first point (pFlag = 0), or at the last one(pFlag =1).
         
         f(x) = Coef[0] + Coef[1] x + Coef[2] x^2 + Coef[3] x^3

       */



        std::vector<T> Coeffs(4,0.0); // Coefficients vector


        int index(0);

        if(pFlag == 1)
        {
            index = 2;
        }

        std::vector<T> matrix(16,0.);
        matrix[0 ] = 1.;
        matrix[1 ] = x[0];
        matrix[2 ] = x[0]*x[0];
        matrix[3 ] = x[0]*x[0]*x[0];

        matrix[4 ] = 1.;
        matrix[5 ] = x[1];
        matrix[6 ] = x[1]*x[1];
        matrix[7 ] = x[1]*x[1]*x[1];

        matrix[8 ] = 1.;
        matrix[9 ] = x[2];
        matrix[10] = x[2]*x[2];
        matrix[11] = x[2]*x[2]*x[2];

        matrix[12] = 0.;
        matrix[13] = 1.;
        matrix[14] = 2.*x[index];
        matrix[15] = 3.*x[index]*x[index];

        std::vector<T> forcing(4,0.);
        forcing[0] = f[0];
        forcing[1] = f[1];
        forcing[2] = f[2];
        forcing[3] = tangent;

        matinv4x4(matrix);

        Coeffs[0] = matrix[0 ]*forcing[0]+matrix[1 ]*forcing[1]+matrix[2 ]*forcing[2]+matrix[3 ]*forcing[3];
        Coeffs[1] = matrix[4 ]*forcing[0]+matrix[5 ]*forcing[1]+matrix[6 ]*forcing[2]+matrix[7 ]*forcing[3];
        Coeffs[2] = matrix[8 ]*forcing[0]+matrix[9 ]*forcing[1]+matrix[10]*forcing[2]+matrix[11]*forcing[3];
        Coeffs[3] = matrix[12]*forcing[0]+matrix[13]*forcing[1]+matrix[14]*forcing[2]+matrix[15]*forcing[3];

        return(Coeffs);


    }
    
    template<class T>
    static std::vector<T>  CubicBulgeCoefs(const std::vector<T> &x, const std::vector<T> &f,T tangent, T curvChangePos, int pFlag)
    {
        /*
         Evaluates the coefficients of a cubic over two points (x[0],f[0]), and (x[1],f[1]),
         specifying a tangent value at the first point (pFlag = 0), or at the last one(pFlag =1),
         and with zero curvature a a certain position curvChangePos E (0,1)*span, measured from
         the point from which the tangent is applied.
         
         f(x) = Coef[0] + Coef[1] x + Coef[2] x^2 + Coef[3] x^3
         
         Calculation:
         
           pFlag = 0:

             f(x)   = d + c (x-x[0]) + b (x-x[0])^2 + a (x-x[0])^3
             f'(x)  = c + 2 b (x-x[0]) + 3 a (x-x[0])^2
             f''(x) = 2 b + 6 a (x-x[0])
             
             d = f[0]

             c = tangent

             a = [ f[1]-f[0] - c (x[1]-x[0]) ]/[  (x[1]-x[0])^3 - 3 curvChangePos (x[1]-x[0])^2  ]
             
             b = - 3 a curvChangePos

           pFlag = 1:

             f(x)   = d + c (x-x[1]) + b (x-x[1])^2 + a (x-x[1])^3
             f'(x)  = c + 2 b (x-x[1]) + 3 a (x-x[1])^2
             f''(x) = 2 b + 6 a (x-x[1])
             
             d = f[1]

             c = tangent

             a = [ f[0]-f[1] - c (x[0]-x[1]) ]/[ CurvChangePos (x[1]-x[0])^2 - (x[1]-x[0])^3]
             
             b =  3 a (x[1]-x[0] - curvChangePos)

       */



        std::vector<T> Coeffs(4,0.0);	// Coefficients vector

        T a(0.0),
                b(0.0),
                c(tangent),
                d(0.0);

        // curvChangePos enters as a nondimensional position between 0 and 1. translate it to a dimensional delta
        // by multiplying it by the span, and perform relevant checks

        if( (curvChangePos < 0.0) || (curvChangePos > 1.0))
            MYTHROW("Problems in utilities::CubicBulgeCoefs(): curvChangePos must be in the (0,1) interval!\n\tcurvChangePos = " + NumTostring(curvChangePos) );

        curvChangePos *= std::abs(x[1]-x[0]);


        switch(pFlag)
        {
        case 0:
        {
            d = f[0];

            a = ( f[1]-f[0] - c*(x[1]-x[0]))/pow(x[1]-x[0],2)/(x[1]-x[0]-3.0*curvChangePos);

            b = -3.0*a*curvChangePos;

            Coeffs[0] = d - c*x[0] + b*x[0]*x[0] - a*x[0]*x[0]*x[0];
            Coeffs[1] = c - 2.*b*x[0] + 3.*a*x[0]*x[0];
            Coeffs[2] = b - 3.*a*x[0];
            Coeffs[3] = a;

            break;
        }
        case 1:
        {
            d = f[1];

            a = ( f[0]-f[0] - c*(x[0]-x[1]))/pow(x[1]-x[0],2)/(3.0*curvChangePos-x[1]+x[0]);

            b = 3.0*a*curvChangePos;

            Coeffs[0] = d - c*x[1] + b*x[1]*x[1] - a*x[1]*x[1]*x[1];
            Coeffs[1] = c - 2.*b*x[1] + 3.*a*x[1]*x[1];
            Coeffs[2] = b - 3.*a*x[1];
            Coeffs[3] = a;

            break;
        }
        default:
        {
            MYTHROW("Problems in utilities::CubicBulgeCoefs(): Wrong pFlag Value!\n");
            break;
        }
        }


        return(Coeffs);


    }
    
    template<class T>
    static T PolyEval(const std::vector<T> &Coeffs, T x, bool der=false)
    {
        /*
          Evaluate a polynomial of arbitrary order, whose coefficients
          are given in the Coef vector, at a certain point.
          
          f(x) = Coef[0] + Sum(Coef[n] x^n), n = 1,...
          
          der = false: Evaluate polynomial
          der = true : Evaluate first derivative
       */

        T poly(0.0);

        for(size_t icoef(1); icoef < Coeffs.size(); ++icoef)
        {
            if(!der)
            {
                poly += Coeffs[icoef]*pow(x,static_cast<T>(icoef));
            }
            else
            {
                poly += static_cast<T>(icoef)*Coeffs[icoef]*pow(x,static_cast<T>(icoef-1));
            }

        }

        // Add the independent coefficient for the non-differentiated case.
        if(!der)
        {
            poly += Coeffs[0];
        }
        return(poly);
    }
    
    template<class T>
    static std::vector<T> PolyEval(const std::vector<T> &Coeffs, const std::vector<T> &x, bool der=false)
    {
        /*
          Evaluate a polynomial of arbitrary order, whose coefficients
          are given in the Coef vector, at a certain mesh
          
          f(x) = Coef[0] + Sum(Coef[n] x^n), n = 1,...
       */

        std::vector<T> poly(0);

        for(size_t j(0); j < x.size(); ++j)
        {
            poly.push_back(PolyEval(Coeffs,x[j], der));
        }

        return(poly);
    }
    

    template<class T>
    static std::vector<T> interpolate(const std::vector<T> &x,
                                      const std::vector<T> &f,
                                      const std::vector<T> &xNew,
                                      const char &method='c')
    {

        /*
    Function interpolation over a new base set. The first and final
    mesh points must be the same

    Method selection:
    - l: Linear interpolation
    - c: Cubic spline interpolation

      */

        size_t N( x.size()    );
        size_t M( xNew.size() );


        if( f.size() != N )
            MYTHROW("Problems in utilities::interpolate(): Number of function points inconsistent with mesh in utilities::interpolate()!\n");

        if(N==0)
            MYTHROW("Problems in utilities::interpolate(): Empty mesh!\n");

        if( (xNew[0] != x[0]) || (xNew[M-1] != x[N-1]) )
            MYTHROW("Problems in utilities::interpolate(): The new mesh corresponds to a different domain!\n");

        std::vector<T> fNew(f);

        tk::spline splineObj;

        if(method=='c')
        {
            splineObj.set_points(x,f);
        }
        else if(method=='l')
        {
            splineObj.set_points(x,f,false);
        }
        else
        {
            MYTHROW("Problems in utilities::interpolate(): Wrong option for interpolation method!\n");
        }


        for(size_t m(1); m < M-1; ++m)
        {
            fNew[m] = splineObj(xNew[m]);
        }


        return(fNew);


    }


    // Translates a set of points to a [0,factor] segment
    template<class T>
    static std::vector<T> NormalizeSet(const std::vector<T> &vec_,const T &factor=1.)
    {

        std::vector<T> vec = vec_;

        size_t siz(vec.size());
        T ini(vec[0]),
          end(vec[siz-1]);

        if(std::abs(ini - end)<1.e3*std::numeric_limits<T>::epsilon())
        {
            MYTHROW("Problems in utilities::NormalizeSet(): The initial point is also the last point! This is not orthodox...\n");
            std::vector<T> dummy(0);return(dummy);
        }

        T idelta(factor/(end-ini));

        for(size_t i(0); i < siz; ++i)
        {
            vec[i] = (vec[i]-ini)*idelta;
        }

        return(vec);

    }

    // Computes the norm of a vector
    template<class T>
    static T ComputeNorm(const std::vector<T> &vec)
    {

        T norm(0.0);

        for(size_t i(0); i < vec.size(); ++i)  norm = vec[i]*vec[i] + norm;

        norm = std::sqrt(norm);

        return(norm);

    }


    template<class T>
    static T ComputeNorm(T *arr,int arrSize)
    {

        T norm(0.0);

        for(size_t i(0); i < arrSize; ++i)  norm = arr[i]*arr[i] + norm;

        norm = std::sqrt(norm);

        return(norm);

    }



    // Writes a matrix as columns to a plain format file
    template<class T>
    static void WriteMatrixToFile(std::string filename, const std::vector< std::vector<T> > &matrix )
    {

        std::ofstream File(filename.c_str());

        size_t M(matrix.size()),
               N(matrix[0].size());


        for(size_t i(0); i < N; ++i)
        {

            for(size_t j(0); j < M; ++j) File << matrix[j][i] << "\t";

            File << std::endl;

        }


        File.close();

    }

    template<class T>
    static std::vector<T> CreateTotalList(const std::vector< std::vector<T> > &SetOfLists)
    {

        /*
    Builds a global list from a set adding elements without repetition

    All the data lists must be previously loaded into "SetOfLists"
      */


        // Initialize with the first list
        std::vector<T> TotalList = SetOfLists[0];

        if(SetOfLists.size() > 1)
        {

            // Add missing data of the remaining lists
            for(size_t i(1); i < SetOfLists.size(); ++i)
            {

                // This sorting operation will slow down the whole thing
                // but its necessary because of the inner workings of binary_search

                std::sort(TotalList.begin(),TotalList.end());

                for(size_t j(0); j < SetOfLists[i].size(); ++j)
                {

                    if(!std::binary_search(TotalList.begin(),TotalList.end(),SetOfLists[i][j]))
                    {
                        TotalList.push_back(SetOfLists[i][j]);
                    }

                }

            }

        }
        
        // Eliminate duplicates

        std::sort(TotalList.begin(),TotalList.end());
        TotalList.erase(std::unique(TotalList.begin(), TotalList.end()), TotalList.end());

        return(TotalList);


    }


    /*-----------------------------------------------------------------------------
  Splits a non bijective curve in upper and lower curves.
  Useful to split in pressure side and suction side variable distributions
--------------------------------------------------------------------------*/

    template<class T>
    static void SplitInTwo(std::vector<T> x_             , std::vector<T> p_          ,
                           std::vector<T> &x_side1       , std::vector<T> &p_side1    ,
                           std::vector<T> &x_side2       , std::vector<T> &p_side2    )
    {
        size_t points = x_.size();
        if(p_.size() != points)
            MYTHROW("Bad arguments in SplitInTwo() function!\n");

        //Split in two
        size_t j_min(0), j_max(0);


        // Look for the maximum and minimum values of the curves
        double x_min, x_max;
        x_min = x_max = x_[0];

        for(size_t j=0; j<points; j++)
        {
            if(x_min > x_[j]) {x_min = x_[j]; j_min = j;}
            if(x_max < x_[j]) {x_max = x_[j]; j_max = j;}
        }

        x_side1.clear();
        x_side2.clear();
        p_side1.clear();
        p_side2.clear();

        if(j_min<j_max)
        {
            for(size_t j=j_min; j<j_max; j++)
            {
                x_side1.push_back(x_[j]);
                p_side1.push_back(p_[j]);
            }

            for(size_t j=j_min; j -->0; )
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
            }

            for(size_t j=points-1; j>=j_max; j--)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
            }
        }
        else
        {
            for(size_t j=j_max; j<j_min; j++)
            {
                x_side1.push_back(x_[j]);
                p_side1.push_back(p_[j]);
            }

            for(size_t j=j_min; j<points; j++)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
            }

            for(size_t j=0; j<=j_max; j++)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
            }
        }
    }


    /*-----------------------------------------------------------------------------
  Overloaded version that splits an associated vector
  Useful to keep track of the working nodes when splitting airfoil distributions
---------------------------------------------------------------------------*/
    template<class T, class T2>
    static void SplitInTwo(std::vector<T> x_             , std::vector<T> p_          ,
                           std::vector<T2> vv_           ,
                           std::vector<T> &x_side1       , std::vector<T> &p_side1    ,
                           std::vector<T2> &vv_side1     ,
                           std::vector<T> &x_side2       , std::vector<T> &p_side2    ,
                           std::vector<T2> &vv_side2     )
    {
        size_t points = x_.size();
        if(p_.size() != points)
            MYTHROW("Bad arguments in SplitInTwo() function!\n");

        //Split in two
        size_t j_min(0), j_max(0);


        // Look for the maximum and minimum values of the curves
        double x_min, x_max;
        x_min = x_max = x_[0];

        for(size_t j=0; j<points; j++)
        {
            if(x_min > x_[j]) {x_min = x_[j]; j_min = j;}
            if(x_max < x_[j]) {x_max = x_[j]; j_max = j;}
        }

        x_side1.clear();
        x_side2.clear();
        p_side1.clear();
        p_side2.clear();
        vv_side1.clear();
        vv_side2.clear();

        if(j_min<j_max)
        {
            for(size_t j=j_min; j<=j_max; ++j)
            {
                x_side1.push_back(x_[j]);
                p_side1.push_back(p_[j]);
                vv_side1.push_back(vv_[j]);
            }

            for(size_t j=j_min; j --> 0;)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
                vv_side2.push_back(vv_[j]);
            }

            for(size_t j=points-1; j>j_max; --j)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
                vv_side2.push_back(vv_[j]);
            }
        }
        else
        {
            for(size_t j=j_max; j<=j_min; ++j)
            {
                x_side1.push_back(x_[j]);
                p_side1.push_back(p_[j]);
                vv_side1.push_back(vv_[j]);
            }

            for(size_t j=j_min; j<points; ++j)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
                vv_side2.push_back(vv_[j]);
            }

            for(size_t j=0; j<j_max; ++j)
            {
                x_side2.push_back(x_[j]);
                p_side2.push_back(p_[j]);
                vv_side2.push_back(vv_[j]);
            }
        }

        size_t SizeCheck(x_side1.size() + x_side2.size());
        size_t sizeDelta(SizeCheck-x_.size());
        while(sizeDelta != 0)
        {
            std::cout << "WARNING: Splitted curves not consistent!" << std::endl
                 << "Delta = " << sizeDelta                    << std::endl
                 << "Popping out the last element of Curve 2!" <<std:: endl;

            x_side2.pop_back();
            p_side2.pop_back();
            vv_side2.pop_back();
            --sizeDelta;
        }

    }

    // Compute the mean value of a vector
    template<class T>
    static T ComputeMean(const std::vector<T> &container)
    {
        size_t N ( container.size() );

        if(N==0) MYTHROW("Empty input in utilities::ComputeMean()!");

        T mean( std::accumulate(container.begin(), container.end(), static_cast<T>(0))/static_cast<T>(N) );

        return mean;
    }

    // Compute the median value of a vector
    template<class T>
    static T ComputeMedian(const std::vector<T> &container)
    {
        size_t N = container.size();

        if(N==0) MYTHROW("Empty input in utilities::ComputeMedian()!");

        T median = container[0];

        if(N>1)
        {
            std::vector<T> sorted(std::sort(container.begin(),container.end()));

            if(isEven(N))
            {
                T mean = sorted[N/2] + sorted[(N/2) - 1];
                median = mean/2;
            }
            else
            {
                median = sorted[N/2];
            }
        }

        return median;
    }

    // Get the standard deviation distribution of a container, and the position of each point in terms of std.
    template<class T1>
    static void ComputeStdDev(const std::vector<T1> &container,std::vector<T1> &stdDevVector,T1 &StdDev, T1 &mean)
    {

        // First compute mean
        size_t N ( container.size() );

        mean = ComputeMean<T1>(container);

        // Now compute the unbiased standard deviation
        StdDev = 0.;
        for(size_t i(0); i < N; ++i) StdDev += (container[i]-mean)*(container[i]-mean);
        StdDev /= static_cast<T1>(N-1);

        StdDev = std::sqrt(StdDev);

        // Finally assign each component a measure in terms of standard deviation
        stdDevVector.clear();
        for(size_t i(0); i < N; ++i)
        {
            stdDevVector.push_back((container[i]-mean)/StdDev);
        }

    }
    template<class T1>
    static void ComputeStdDev(const std::vector<T1> &container,std::vector<T1> &stdDevVector,T1 &StdDev)
    {
       T1 mean(0.);
       ComputeStdDev<T1>(container,stdDevVector,StdDev,mean);
    }

    // Get the maximum, minimum and maximum absolute value
    template<class T1, class T2>
    static void ContainerStats(const std::vector<T1> &container,T2 &min,T2 &max, T2 &maxabs)
    {

        max = *std::max_element(container.begin(),container.end());
        min = *std::min_element(container.begin(),container.end());


        if(std::abs(max) <= std::abs(min))
            maxabs = std::abs(min);
        else
            maxabs = std::abs(max);

    }

    template<class T1, class T2>
    static void ContainerStats(const std::vector< std::vector<T1> > &container,T2 &min,T2 &max, T2 &maxabs)
    {

        std::vector< std::vector<T1> > dummy(3);
        dummy[0].resize(container.size()); // Will hold the maximum values of each vector in container
        dummy[1].resize(container.size()); // The minimum
        dummy[2].resize(container.size()); // The maximum absolute value

        for(size_t i(0); i <  container.size(); ++i)
        {
            ContainerStats(container[i],dummy[0][i],dummy[1][i],dummy[2][i]);
        }

        max    = *std::max_element(dummy[0].begin(),dummy[0].end());
        min    = *std::min_element(dummy[1].begin(),dummy[1].end());
        maxabs = *std::max_element(dummy[2].begin(),dummy[2].end());



    }

    template<class T>
    static int findIdx(const std::vector< std::vector<T> > &container, T objective, int &idx1, int &idx2)
    {
        /*
 *
 * 	Auxiliary function to find an element and its indices in a 2nd order vector
 *
 * 	Could be recursively extended for an input container order, but for now...
 *
 */

        idx1 = 0;
        idx2 = 0;
        int exitFlag(-1);


        for(size_t i(0); i < container.size(); ++i)
        {

            for(size_t j(0); j < container[i].size(); ++j)
            {

                if(std::abs(container[i][j] - objective)<1.e3*std::numeric_limits<T>::epsilon())
                {
                    idx1 = i;
                    idx2 = j;
                    exitFlag = 0;
                    return(exitFlag);

                }

            }


        }

        return(exitFlag);
    }

    template<class T>
    static size_t findNearest(const std::vector<T> &container, T objective)
    {

        std::vector<T> distance(0);

        size_t cSiz(container.size());

        for(size_t i(0); i < cSiz; ++i)
        {
            distance.push_back( std::abs(container[i]-objective) );
        }

        size_t idx ( std::min_element(distance.begin(), distance.end()) - distance.begin() );


        return( idx );


    }


    template <class T>
    static bool binary_search_pos ( std::vector<T> &vv, const T& value , size_t &idx)
    {
        /*
   *
   *  Binary search that gives the position of "vv" where "value" has been found
   *
   */


        typedef class std::vector<T>::iterator iter;
        iter first;

        first = std::lower_bound(vv.begin(),vv.end(),value);
        idx = size_t( first - vv.begin() );
        return (first!=vv.end() && !(value<*first));
    }


    template <class T>
    static int round ( T toRound)
    {
        /*
   *
   *  Round a floating point number to nearest integer
   *
   *  Halfway values always get rounded up
   *
   */


        return( floor(toRound + 0.5) );

    }

    /*
     *
     *  Function that replaces a pattern of a string with another.
     *
     */
    static bool replace(std::string& str, const std::string& from, const std::string& to);

    template<class T>
    static void removeEmpty(std::vector<std::vector<T>> &vector)
    {
        for (size_t i = 0; i < vector.size(); )
        {
            if (vector[i].size() == 0) {
                vector.erase(vector.begin() + i);
            } else ++i;
        }
    }


    // Find the indexes of the peaks of a signal which are above a certain threshold
    template<class T>
    static std::vector<size_t> find_peaks(const std::vector<T> &signal, const T &Threshold=1.e-6)
    {

        auto appendIfAboveThreshold = [Threshold](size_t &iniIdx,const std::vector<T> &signal_)
        {
            std::vector<T> crop(0);
            T relTh = Threshold*(*std::max_element(signal_.begin(),signal_.end()));

            bool record(false);

            while(iniIdx<signal_.size())
            {
                T si = signal_[iniIdx];

                if(si > relTh)
                {
                    record=true;
                }

                if(record)
                {
                    crop.push_back(si);

                    if(si < relTh)
                    {
                        record = false;
                        break;
                    }
                }

                ++iniIdx;
            }
            return crop;
        };

        std::vector<std::vector<T>> peakList(0);

        size_t idx=0;
        while(idx<signal.size())
        {
            // Idx is updates inside appendIfAbovethreshold
            peakList.push_back(appendIfAboveThreshold(idx,signal));
        }

        removeEmpty<T>(peakList);

        std::vector<T> maxList(0);
        for(size_t i(0); i < peakList.size(); ++i)
        {
           maxList.push_back(*std::max_element(peakList[i].begin(),peakList[i].end()));
        }

        std::vector<size_t> peakIdx(0);
        for(size_t i(0); i < peakList.size(); ++i)
        {
            size_t foundIdx = searchList<T>(signal,maxList[i])[0];
            peakIdx.push_back(foundIdx);
        }

        FilterDuplicates<size_t>(peakIdx);

        return peakIdx;
    }


/*
 *
 *  Perform the 1D real FFT of a single period signal stored in a vector container.
 *  The extended signal and the new basis overwrite the input values.
 *  The FFT is given over the frequency vector freq.
 *
 *  mode = 0 -> Direct transform.
 *  mode = 1 -> Inverse transform.
 *
 *  rep = 0 -> Output given as Power spectrum - Phase
 *  rep = 1 -> Output given as Re(F)- Im(F)(stored in the phase container)
 *
 */

    template <class T>
    static void FFT_R1D(std::vector<T> &signal,
                        std::vector<T> &basis,
                        std::vector<T> &Transform,
                        std::vector<T> &phase,
                        std::vector<T> &freq,
                        const short int &mode,
                        const short int &rep)
    {

        std::vector<T> newSignal(0);
        std::vector<T> newBasis (0);

        if(mode == 0)
        {
            if(signal.size() != basis.size())
                MYTHROW("Signal and basis vectors are not consistent!\n");

            // Interpolate linearly over a uniform basis whose size is the next power of 2.

            size_t Size(signal.size());
            T L(basis.back()-basis[0]),
              dx( L/static_cast<T>(Size-1) );

            if(!isPowerOfTwo(Size))
            {
                Size = std::pow(2, ceil(log(Size)/log(2)));

                dx = L/static_cast<T>(Size-1);

                newBasis .push_back(basis [0]);
                for(size_t i(1); i < Size - 1 ;++i)
                {
                    newBasis.push_back(newBasis[i-1] + dx);
                }
                newBasis .push_back(basis .back());

                newSignal = interpolate(basis,signal,newBasis,'c');
            }
            else
            {
                newSignal = signal;
                newBasis  = basis;
            }
            // Perform the FFT
            T fsamp(TWOPI/dx),
              df(fsamp/static_cast<T>(Size-1));


            realft<T>(newSignal,1);

            for(size_t i(0); i < 2*Size; ++i)
                newSignal[i] /= static_cast<T>(Size);

            Transform.clear();
            phase.clear();
            freq.clear();

            T modul(0),arg(0);

            for(size_t i(0); i < Size; ++i)
            {

                freq.push_back(df*i);

                if(rep == 0)
                {
                    modul = std::sqrt (newSignal[2*i + 0]*newSignal[2*i + 0] + newSignal[2*i + 1]*newSignal[2*i + 1]);
                    arg   = std::atan2(newSignal[2*i + 1],newSignal[2*i + 0]);

                    Transform.push_back(modul);
                    phase    .push_back(arg);
                }
                else if(rep == 1)
                {
                    Transform.push_back(newSignal[2*i + 0]);
                    phase    .push_back(newSignal[2*i + 1]);
                }

            }

        }
        else if(mode == 1)
        {

            if(Transform.size() != phase.size())
            {
                MYTHROW("Transform and phase vectors not consistent in FFT_R1D!\n")
            }

            size_t nn(2*Transform.size());

            T Real(0),Imag(0);

            for(size_t i(0); i < Transform.size(); ++i)
            {

                if(rep == 0)
                {
                    Real = Transform[i]*cos(phase[i]);
                    Imag = Transform[i]*sin(phase[i]);

                    newSignal.push_back(Real);
                    newSignal.push_back(Imag);
                }
                else if(rep == 1)
                {
                    newSignal.push_back(Transform[i]);
                    newSignal.push_back(phase[i]);
                }

            }


            // Perform the IFFT
            realft<T>(newSignal,-1);

            for(size_t i(0); i < newSignal.size(); ++i)
            {
                newSignal[i] *= static_cast<T>(nn)/2.;
            }

            // Build the basis for the reconstructed signal.
            size_t Size(newSignal.size());
            T dx( std::abs(basis.back()-basis[0]) );
              dx /= static_cast<T>(Size-1);


            for(size_t i(0); i < Size; ++i)
            {
                newBasis.push_back(basis[0]+static_cast<T>(i)*dx );
            }

            signal = interpolate(newBasis,newSignal,basis,'c');

        }
        else
        {
            MYTHROW("Wrong mode value in utilities::FFT_R1D()!\n");
        }


    }



    /*
 *
 * Calls FFT_R1D and postprocesses the output, so that a vector with only the harmonic content is given.
 * The output is the one sided spectrum, given for Nharm multiples of the natural frequency. The mean value
 * is stored in the first element.
 *
 * If the spectrum is expected to be discrete with clearly separate peaks, the peaks are extracted
 *
 * NOTE: Assumes a proper period is used so that the peaks are discrete. If the signal is not periodic, this will fail,
 *       as the peaks will be smeared out (spectral leakage). You don want this anyway if you have leakage.
 *
 *  rep = 0 -> Output given as Power spectrum - Phase
 *  rep = 1 -> Output given as Re(F)- Im(F)(stored in the phase container)
 */
    template <class T>
    static void Spectrum_R1D(std::vector<T> &signal,
                             std::vector<T> &basis,
                             std::vector<T> &Transform,
                             std::vector<T> &phase,
                             std::vector<T> &freq,
                             const size_t &Nharm,
                             short int &mode,
                             short int &rep,
                             bool discrete=true)
    {
        if(mode==0)
        {

            Transform.clear();
            phase.clear();
            freq.clear();

            std::vector<T> fullTransform(0),
                           fullphase(0),
                           fullfreq(0);

            // Compute the mean and substract it
            size_t N = signal.size();
            T mean = ComputeMean<T>(signal);


            std::vector<T> signalWithoutMean(signal);
            for(size_t i(0); i < N; ++i) signalWithoutMean[i]-=mean;

            FFT_R1D<T>(signalWithoutMean,basis,fullTransform,fullphase,fullfreq,mode,rep);

            if(rep ==0)
            {
                for(size_t idx(0); idx < fullTransform.size(); ++idx)
                {
                    fullTransform[idx] *= 2;
                }
            }

            std::vector<T> modulus = fullTransform;
            if(rep == 1)
            {
                for(size_t idx(0); idx < fullTransform.size(); ++idx)
                {
                    fullTransform[idx] *= 2.;
                    fullphase[idx]     *= 2.;
                    modulus[idx] = std::sqrt(fullTransform[idx]*fullTransform[idx]+fullphase[idx]*fullphase[idx]);
                }
            }

            Transform.push_back(mean);
            phase.push_back(0.);
            freq.push_back(0.);

            if(discrete)
            {

                std::vector<size_t> peaks = find_peaks<T>(modulus,1.e-2);
                size_t nPeaks   = peaks.size(),
                       nHarmMod = std::min(Nharm,nPeaks);

                for(size_t harm(0); harm < nHarmMod; ++harm)
                {
                    size_t idx = peaks[harm];
                    Transform.push_back(fullTransform[idx]);
                    phase.push_back(fullphase[idx]);
                    freq.push_back(fullfreq[idx]);
                }
            }
            else
            {
                Transform.insert(Transform.end(), fullTransform.begin()+1,fullTransform.begin()+Nharm+1);
                phase.insert(phase.end(), fullphase.begin()+1,fullphase.begin()+Nharm+1);
                freq.insert(freq.end(), fullfreq.begin()+1,fullfreq.begin()+Nharm+1);
            }

        }
        else if(mode == 1)
        {
            signal.clear();

            size_t nHarmMod = std::min(Nharm,Transform.size()-1);

            if(!isCloseToZero(freq[0])) MYTHROW("First frequency is not 0 in utilities::Spectrum_R1D()");
            if(freq.size() != nHarmMod+1) MYTHROW("Frequency vector not consistent with Fourier series in utilities::Spectrum_R1D()");

            // Compute the periodic expansion
            for(size_t p(0); p < basis.size(); ++p)
            {
                T expansion(Transform[0]);
                for(size_t harm(1); harm < nHarmMod+1; ++harm)
                {
                    T Real(Transform[harm]),Imag(Transform[harm]);

                    if(rep == 0)
                    {
                        Real *= std::cos(phase[harm]);
                        Imag *= std::sin(phase[harm]);
                    }
                    else if(rep == 1)
                    {
                        Imag = phase[harm];
                    }
                    T F = freq[harm],
                      x = basis[p]-basis[0];

                    expansion += Real*std::cos(F*x)-Imag*std::sin(F*x);
                }
                signal.push_back(expansion);
            }

        }
        else
        {
            MYTHROW("Wrong mode in utilities::Spectrum_R1D()");
        }

    }



    // Taken from https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
    // CooleyTukey FFT (in-place, divide-and-conquer)
    // Higher memory requirements and redundancy although more intuitive
    template<class T>
    static void fft(std::valarray<std::complex<T>> &x)
    {
        typedef std::complex<T> Complex;
        typedef std::valarray<Complex> CArray;

        const size_t N = x.size(); // N must be a power of 2
        if (N <= 1) return;

        // divide
        CArray even = x[std::slice(0, N/2, 2)];
        CArray  odd = x[std::slice(1, N/2, 2)];

        // conquer
        fft(even);
        fft(odd);

        // combine
        for (size_t k = 0; k < N/2; ++k)
        {
            Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
            x[k    ] = even[k] + t;
            x[k+N/2] = even[k] - t;
        }
    }

    // inverse fft (in-place)
    template<class T>
    static void ifft(std::valarray<std::complex<T>>& x)
    {
        // conjugate the complex numbers
        x = x.apply(std::conj);

        // forward fft
        fft( x );

        // conjugate the complex numbers again
        x = x.apply(std::conj);

        // scale the numbers
        x /= x.size();
    }


    // Do the fft of a real signal
    // Input:
    //   if dir = 1, FFT
    //   x: One real signal
    // Output:
    //   x: Transform of x in a double sized vector, x[2*i] = Re(X), x[2*i+1] = Im(X)
    // Input:
    //   if dir = -1, IFFT
    //   x: Transform of x given in a double sized vector, x[2*i] = Re(X), x[2*i+1] = Im(X)
    // Output:
    //   x: One real signal
    template<class T>
    static void realft(std::vector<T> &x_,const short int &dir)
    {
        typedef std::complex<T> Complex;
        typedef std::valarray<Complex> CArray;

        size_t nn( x_.size());

        std::vector<Complex> Cx(0);


        if(dir == 1)
        {
            for(size_t idx(0); idx < nn; ++idx)
            {
                Complex cn(x_[idx],0.);
                Cx.push_back(cn);
            }

            CArray Z(Cx.data(),Cx.size());

            fft<T>(Z);

            x_.clear();
            for(size_t idx(0); idx < nn; ++idx)
            {
                Complex z = Z[idx];

                x_.push_back(z.real());
                x_.push_back(z.imag());
            }
        }
        else if(dir == -1)
        {
            for(size_t idx(0); idx < nn/2; ++idx)
            {
                Complex cn(x_[2*idx],x_[2*idx+1]);
                Cx.push_back(cn);
            }

            CArray Z(Cx.data(),Cx.size());

            ifft<T>(Z);

            x_.clear();
            for(size_t idx(0); idx < nn/2; ++idx)
            {
                Complex z = Z[idx];

                x_.push_back(z.real());
            }
        }
        else
        {
            MYTHROW("Wrong dir in utilities::realfft()");
        }


    }


    /*
 *
 *  Compute the absolute value of the curvature of a smooth (not noisy) 1D curve using the osculating circle method
 *
 *   Input:
 *       data: The discrete curve points.
 *       base: The discrete base mesh.
 *       isClosedCurve: considers a closed and periodic curve.
 *
 */
    template<class T,class T2>
    static std::vector<T> computeSmoothCurveCurvature(const std::vector<T> &data, const std::vector<T2> &base,bool isClosedCurve=false,const T &fac_eps=1.e4)
    {
        size_t dataSiz(data.size());

        if(dataSiz!=base.size())MYTHROW("Problems in utilities::computeCurveDerivatives3(): Inconsistent input!\n");

        std::vector<T> curvature(0);
        double eps = fac_eps*std::numeric_limits<T>::epsilon();

        for(size_t idx(0); idx < dataSiz;++idx)
        {

            std::vector<T> localData(3,0.),localBase(3,0.);

            if(idx==0 && isClosedCurve)
            {
                localBase[0] = base[dataSiz-1];
                localBase[1] = base[idx  ];
                localBase[2] = base[idx+1];

                localData[0] = data[dataSiz-1];
                localData[1] = data[idx  ];
                localData[2] = data[idx+1];
            }
            else if(idx==0 && !isClosedCurve)
            {
                localBase[0] = base[idx  ];
                localBase[1] = base[idx+1];
                localBase[2] = base[idx+2];

                localData[0] = data[idx  ];
                localData[1] = data[idx+1];
                localData[2] = data[idx+2];
            }
            else if(idx==dataSiz-1 && isClosedCurve)
            {
                localBase[0] = base[idx-1];
                localBase[1] = base[idx  ];
                localBase[2] = base[0    ];

                localData[0] = data[idx-1];
                localData[1] = data[idx  ];
                localData[2] = data[0    ];
            }
            else if(idx==dataSiz-1 && !isClosedCurve)
            {
                localBase[0] = base[idx-2];
                localBase[1] = base[idx-1];
                localBase[2] = base[idx  ];

                localData[0] = data[idx-2];
                localData[1] = data[idx-1];
                localData[2] = data[idx  ];
            }
            else
            {
                localBase[0] = base[idx-1];
                localBase[1] = base[idx  ];
                localBase[2] = base[idx+1];

                localData[0] = data[idx-1];
                localData[1] = data[idx  ];
                localData[2] = data[idx+1];
            }


            double aa     = localData[1]*localData[1]-localData[0]*localData[0]+localBase[1]*localBase[1]-localBase[0]*localBase[0],
                   bb     = localData[2]*localData[2]-localData[0]*localData[0]+localBase[2]*localBase[2]-localBase[0]*localBase[0],
                   delta1 = localBase[1]-localBase[0],
                   delta2 = localBase[2]-localBase[0],
                   a      = aa/delta1,
                   b      = bb/delta2,
                   c      = (localData[1]-localData[0])/delta1,
                   d      = (localData[2]-localData[0])/delta2;


            double y0 = 0.5*(a-b)/(c-d),
                   x0 = 0.5*a-c*y0;

            double dx = localBase[0]-x0,
                   dy = localData[0]-y0,
                   R  = std::sqrt(dx*dx+dy*dy);

            double curv = 0;

            if(R > eps)
            {
                curv = 1./R;
            }
            else
            {
                curv = 1./eps;
            }

            curvature.push_back(curv);

        }

        return curvature;

    }

    /*
 *
 *  Compute the derivatives of a discrete curve using a three point stencil.
 *
 *   Input:
 *       data: The discrete curve points.
 *       base: The discrete base mesh.
 *       order: The order of the asked for derivatives, 1 to get the slope distribution, 2 to get the second derivative distribution, -1 to get the signed curvature distribution, -2 to get the absolute value of the curvature
 *       isClosedCurve: considers a closed and periodic curve.
 */
    template<class T,class T2>
    static std::vector<T> computeCurveDerivatives3(const std::vector<T> &data, const std::vector<T2> &base, int order,bool isClosedCurve=false)
    {

        if(data.size()!=base.size())MYTHROW("Problems in utilities::computeCurveDerivatives3(): Inconsistent input!\n");

        if(order > 2 || order < -2)MYTHROW("Problems in utilities::computeCurveDerivatives3(): Wrong order!\n");


        std::vector<T> out(data.size(),0.),
                 localData(3,0.);

        std::vector<T2> localBase(3,0.);

        size_t sm1(data.size()-1);

        // Process the interior domain
        for(size_t point(1); point < sm1; ++point)
        {
            localData[0] = data[point-1];
            localData[1] = data[point  ];
            localData[2] = data[point+1];

            localBase[0] = base[point-1];
            localBase[1] = base[point  ];
            localBase[2] = base[point+1];

            if(order == 1)
            {
                out[point] = threePointFirstDerivative(localData,localBase)[1];
            }
            else if(order == 2)
            {
                out[point] = threePointSecondDerivative(localData,localBase)[1];
            }
            else if(order < 0)
            {
                T der  = threePointFirstDerivative(localData,localBase)[1],
                  der2 = threePointSecondDerivative(localData,localBase)[1],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[point] = curvature;
            }
        }

        // Process the boundaries
        if(isClosedCurve)
        {
            localData[0] = data[sm1];
            localData[1] = data[0];
            localData[2] = data[1];

            localBase[0] = base[sm1];
            localBase[1] = base[0];
            localBase[2] = base[1];

            if(order == 1)
            {
                out[0] = threePointFirstDerivative(localData,localBase)[1];
            }
            else if(order == 2)
            {
                out[0] = threePointSecondDerivative(localData,localBase)[1];
            }
            else if(order < 0)
            {
                T der  = threePointFirstDerivative(localData,localBase)[1],
                  der2 = threePointSecondDerivative(localData,localBase)[1],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[0] = curvature;
            }

            localData[0] = data[sm1-1];
            localData[1] = data[sm1];
            localData[2] = data[0];

            localBase[0] = base[sm1-1];
            localBase[1] = base[sm1];
            localBase[2] = base[0];

            if(order == 1)
            {
                out[sm1] = threePointFirstDerivative(localData,localBase)[1];
            }
            else if(order == 2)
            {
                out[sm1] = threePointSecondDerivative(localData,localBase)[1];
            }
            else if(order < 0)
            {
                T der  = threePointFirstDerivative(localData,localBase)[1],
                  der2 = threePointSecondDerivative(localData,localBase)[1],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[sm1] = curvature;
            }
        }
        else
        {
            localData[0] = data[0];
            localData[1] = data[1];
            localData[2] = data[2];

            localBase[0] = base[0];
            localBase[1] = base[1];
            localBase[2] = base[2];

            if(order == 1)
            {
                out[0] = threePointFirstDerivative(localData,localBase)[0];
            }
            else if(order == 2)
            {
                out[0] = threePointSecondDerivative(localData,localBase)[0];
            }
            else if(order < 0)
            {
                T der  = threePointFirstDerivative(localData,localBase)[0],
                  der2 = threePointSecondDerivative(localData,localBase)[0],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[0] = curvature;
            }

            localData[0] = data[sm1-2];
            localData[1] = data[sm1-1];
            localData[2] = data[sm1];

            localBase[0] = base[sm1-2];
            localBase[1] = base[sm1-1];
            localBase[2] = base[sm1];

            if(order == 1)
            {
                out[sm1] = threePointFirstDerivative(localData,localBase)[2];
            }
            else if(order == 2)
            {
                out[sm1] = threePointSecondDerivative(localData,localBase)[2];
            }
            else if(order < 0)
            {
                T der  = threePointFirstDerivative(localData,localBase)[2],
                  der2 = threePointSecondDerivative(localData,localBase)[2],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[sm1] = curvature;
            }
        }

        return(out);

    }


    /*
 *
 *  Particularization of computeCurveDerivatives(const std::vector<T> &data, const std::vector<T> &base, int order,bool isClosedCurve)
 *  for an evenly spaced mesh.
 *
 */
    template<class T,class T2>
    static std::vector<T> computeCurveDerivatives3(const std::vector<T> &data, T2 meshSize, int order,bool isClosedCurve=false)
    {

        std::vector<T2> base(0);

        for(size_t point(0); point < data.size(); ++point) base.push_back(point*meshSize);

        return(computeCurveDerivatives3(data,base,order,isClosedCurve));

    }

    /*
 *
 *  Compute the derivatives of a discrete curve using a five point stencil.
 *
 *   Input:
 *       data: The discrete curve points.
 *       base: The discrete base mesh.
 *       order: The order of the asked for derivatives, 1 to get the slope distribution, 2 to get the curvature distribution.
 *       isClosedCurve: considers a closed and periodic curve.
 */
    template<class T,class T2>
    static std::vector<T> computeCurveDerivatives5(const std::vector<T> &data, const std::vector<T2> &base, int order,bool isClosedCurve)
    {

        if(data.size()!=base.size())MYTHROW("Problems in utilities::computeCurveDerivatives5(): Inconsistent input!\n");
        if(order > 2 || order < -2)MYTHROW("Problems in utilities::computeCurveDerivatives3(): Wrong order!\n");

        std::vector<T> out(data.size(),0.),
                  localData(5,0.),
                  localData4(4,0.);

        std::vector<T2> localBase(5,0.),
                   localBase4(4,0.);


        // Process the interior domain
        for(size_t point(2); point < data.size()-2; ++point)
        {
            localData[0] = data[point-2];
            localData[1] = data[point-1];
            localData[2] = data[point  ];
            localData[3] = data[point+1];
            localData[4] = data[point+2];

            localBase[0] = base[point-2];
            localBase[1] = base[point-1];
            localBase[2] = base[point  ];
            localBase[3] = base[point+1];
            localBase[4] = base[point+2];

            if(order == 1)
            {
                out[point] = fivePointFirstDerivative(localData,localBase)[2];
            }
            else if(order == 2)
            {
                out[point] = fivePointSecondDerivative(localData,localBase)[2];
            }
            else if(order < 0)
            {
                T der  = fivePointFirstDerivative(localData,localBase)[2],
                  der2 = fivePointSecondDerivative(localData,localBase)[2],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[point] = curvature;
            }
        }

        // Process the boundaries
        if(isClosedCurve)
        {
            // First point
            localData[0] = data[data.size()-2];
            localData[1] = data[data.size()-1];
            localData[2] = data[0];
            localData[3] = data[1];
            localData[4] = data[2];

            localBase[0] = base[data.size()-2];
            localBase[1] = base[data.size()-1];
            localBase[2] = base[0];
            localBase[3] = base[1];
            localBase[4] = base[2];

            if(order == 1)
            {
                out[0] = fivePointFirstDerivative(localData,localBase)[2];
            }
            else if(order == 2)
            {
                out[0] = fivePointSecondDerivative(localData,localBase)[2];
            }
            else if(order < 0)
            {
                T der  = fivePointFirstDerivative(localData,localBase)[2],
                  der2 = fivePointSecondDerivative(localData,localBase)[2],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[0] = curvature;
            }

            // Second point
            localData[0] = data[data.size()-1];
            localData[1] = data[0];
            localData[2] = data[1];
            localData[3] = data[2];
            localData[4] = data[3];

            localBase[0] = base[data.size()-1];
            localBase[1] = base[0];
            localBase[2] = base[1];
            localBase[3] = base[2];
            localBase[4] = base[3];

            if(order == 1)
            {
                out[1] = fivePointFirstDerivative(localData,localBase)[2];
            }
            else if(order == 2)
            {
                out[1] = fivePointSecondDerivative(localData,localBase)[2];
            }
            else if(order < 0)
            {
                T der  = fivePointFirstDerivative(localData,localBase)[2],
                  der2 = fivePointSecondDerivative(localData,localBase)[2],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[1] = curvature;
            }



            // Second to last point

            localData[0] = data[data.size()-4];
            localData[1] = data[data.size()-3];
            localData[2] = data[data.size()-2];
            localData[3] = data[data.size()-1];
            localData[4] = data[0];

            localBase[0] = base[data.size()-4];
            localBase[1] = base[data.size()-3];
            localBase[2] = base[data.size()-2];
            localBase[3] = base[data.size()-1];
            localBase[4] = base[0];

            if(order == 1)
            {
                out[data.size()-2] = fivePointFirstDerivative(localData,localBase)[2];
            }
            else if(order == 2)
            {
                out[data.size()-2] = fivePointSecondDerivative(localData,localBase)[2];
            }
            else if(order < 0)
            {
                T der  = fivePointFirstDerivative(localData,localBase)[2],
                  der2 = fivePointSecondDerivative(localData,localBase)[2],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[data.size()-2] = curvature;
            }

            // Last point

            localData[0] = data[data.size()-3];
            localData[1] = data[data.size()-2];
            localData[2] = data[data.size()-1];
            localData[3] = data[0];
            localData[4] = data[1];

            localBase[0] = base[data.size()-3];
            localBase[1] = base[data.size()-2];
            localBase[2] = base[data.size()-1];
            localBase[3] = base[0];
            localBase[4] = base[1];

            if(order == 1)
            {
                out[data.size()-1] = fivePointFirstDerivative(localData,localBase)[2];
            }
            else if(order == 2)
            {
                out[data.size()-1] = fivePointSecondDerivative(localData,localBase)[2];
            }
            else if(order < 0)
            {
                T der  = fivePointFirstDerivative(localData,localBase)[2],
                  der2 = fivePointSecondDerivative(localData,localBase)[2],
                  curvature = der2/pow(1.+der*der,1.5);

                if(order == -2)
                {
                    curvature = std::abs(curvature);
                }

                out[data.size()-1] = curvature;
            }
        }
        else
        {

            // First point
            localData[0] = data[0];
            localData[1] = data[1];
            localData[2] = data[2];
            localData[3] = data[3];
            localData[4] = data[4];

            localBase[0] = base[0];
            localBase[1] = base[1];
            localBase[2] = base[2];
            localBase[3] = base[3];
            localBase[4] = base[4];

            if(order == 1)
            {
                out[0] = fivePointFirstDerivative(localData,localBase)[0];
            }
            else if(order == 2)
            {
                out[0] = fivePointSecondDerivative(localData,localBase)[0];
            }


            // Second point: Using a four point stencil! See fivePointFirstDerivative and fivePointSecondDerivative for the reason of it.

            localData4[0] = data[0];
            localData4[1] = data[1];
            localData4[2] = data[2];
            localData4[3] = data[3];

            localBase4[0] = base[0];
            localBase4[1] = base[1];
            localBase4[2] = base[2];
            localBase4[2] = base[3];


            if(order == 1)
            {
                // The first derivative can be computed correctly with a five point stencil.
                out[1] = fivePointFirstDerivative(localData,localBase)[1];
            }
            else if(order == 2)
            {
                out[1] = fourPointSecondDerivative(localData4,localBase4)[1];
            }


            // Second to last point: Again, four point stencil

            localData4[0] = data[data.size()-4];
            localData4[1] = data[data.size()-3];
            localData4[2] = data[data.size()-2];
            localData4[3] = data[data.size()-1];

            localBase4[0] = base[data.size()-4];
            localBase4[1] = base[data.size()-3];
            localBase4[2] = base[data.size()-2];
            localBase4[3] = base[data.size()-1];


            if(order == 1)
            {
                out[data.size()-2] = fourPointFirstDerivative(localData4,localBase4)[2];
            }
            else if(order == 2)
            {
                out[data.size()-2] = fourPointSecondDerivative(localData4,localBase4)[2];
            }

            // Last point

            localData[0] = data[data.size()-5];
            localData[1] = data[data.size()-4];
            localData[2] = data[data.size()-3];
            localData[3] = data[data.size()-2];
            localData[4] = data[data.size()-1];

            localBase[0] = base[data.size()-5];
            localBase[1] = base[data.size()-4];
            localBase[2] = base[data.size()-3];
            localBase[3] = base[data.size()-2];
            localBase[4] = base[data.size()-1];

            if(order == 1)
            {
                out[data.size()-1] = fivePointFirstDerivative(localData,localBase)[4];
            }
            else if(order == 2)
            {
                out[data.size()-1] = fivePointSecondDerivative(localData,localBase)[4];
            }
        }

        return(out);

    }


    /*
 *
 *  Particularization of computeCurveDerivatives(const std::vector<T> &data, const std::vector<T> &base, int order,bool isClosedCurve)
 *  for an evenly spaced mesh.
 *
 */
    template<class T, class T2>
    static std::vector<T> computeCurveDerivatives5(const std::vector<T> &data, T2 meshSize, int order,bool isClosedCurve=false)
    {

        std::vector<T2> base(0);

        for(size_t point(0); point < data.size(); ++point) base.push_back(point*meshSize);

        return(computeCurveDerivatives5(data,base,order,isClosedCurve));

    }

    /*
 *
 *  Auxiliar interpolation algorithms used in the computeCurveDerivatives methods.
 *  The output is the value of the derivatives at each mesh point,
 *
 *  Reference:
 *
 *     "Finite Difference Formulae for Unequal Sub-Intervals Using Lagrange's Interpolation Formula"
 *      A.K. Singh & B.S. Bhadauria
 *     Int. Journal of Math. Analysis, Vol 3, 2009, number 17, 815-827
 *
 */
    template<class T,class T2>
    static std::vector<T> threePointFirstDerivative(const std::vector<T> &data, const std::vector<T2> &base)
    {

        if(data.size()!=base.size())MYTHROW("Problems in utilities::threePointFirstDerivative(): Inconsistent input!\n");
        if(data.size()!=3)MYTHROW("Problems in utilities::threePointFirstDerivative(): Input not of size 3!\n");

        std::vector<T> out(data.size(),0.),
                  coeffs(data.size(),0.);

        std::vector<T2> delta(0);

        for(size_t point(0); point < data.size()-1; ++point) delta.push_back(base[point+1]-base[point]);

        coeffs[0] = -(2.*delta[0]+delta[1])/(delta[0]*(delta[0]+delta[1]));
        coeffs[1] = (delta[0]+delta[1])/(delta[0]*delta[1]);
        coeffs[2] = -delta[1]/(delta[1]*(delta[0]+delta[1]));

        out[0] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2];

        coeffs[0] = -delta[1]/(delta[0]*(delta[0]+delta[1]));
        coeffs[1] = -(delta[0]-delta[1])/(delta[0]*delta[1]);
        coeffs[2] = delta[0]/(delta[1]*(delta[0]+delta[1]));

        out[1] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2];

        coeffs[0] = delta[1]/(delta[0]*(delta[0]+delta[1]));
        coeffs[1] = -(delta[0]+delta[1])/(delta[0]*delta[1]);
        coeffs[2] = (delta[0]+2.*delta[1])/(delta[1]*(delta[0]+delta[1]));

        out[2] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2];


        return(out);

    }

    template<class T,class T2>
    static std::vector<T> threePointSecondDerivative(const std::vector<T> &data, const std::vector<T2> &base)
    {

        if(data.size()!=base.size())MYTHROW("Problems in utilities::threePointSecondDerivative(): Inconsistent input!\n");
        if(data.size()!=3)MYTHROW("Problems in utilities::threePointSecondDerivative(): Input not of size 3!\n");

        std::vector<T> out(data.size(),0.),
                  coeffs(data.size(),0.);

        std::vector<T2> delta(0);

        for(size_t point(0); point < data.size()-1; ++point) delta.push_back(base[point+1]-base[point]);

        T2 factor( 2./(delta[0]*delta[1]*(delta[0]+delta[1])) );

        coeffs[0] =   delta[1]*factor;
        coeffs[1] = -(delta[0]+delta[1])*factor;
        coeffs[2] =   delta[0]*factor;

        out[0] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2];

        out[1] = out[0];
        out[2] = out[0];


        return(out);

    }

    template<class T,class T2>
    static std::vector<T> fourPointFirstDerivative(const std::vector<T> &data, const std::vector<T2> &base)
    {
        if(data.size()!=base.size())MYTHROW("Problems in utilities::fourPointFirstDerivative(): Inconsistent input!\n");
        if(data.size()!=4)MYTHROW("Problems in utilities::fourPointFirstDerivative(): Input not of size 4!\n");

        std::vector<T> out(data.size(),0.),
                  coeffs(data.size(),0.);

        std::vector<T2> delta(0);

        for(size_t point(0); point < data.size()-1; ++point) delta.push_back(base[point+1]-base[point]);

        T2 sumDelta(base.end()-base.begin());

        coeffs[0] = -((2.*delta[0]+delta[1])*sumDelta+delta[0]*(delta[0]+delta[1]))/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = (sumDelta*(delta[0]+delta[1]))/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = -delta[0]*sumDelta/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = delta[0]*(delta[0]+delta[1])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[0] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = -delta[1]*(delta[1]+delta[2])/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = (delta[1]*(delta[1]+delta[2])-delta[0]*(2.*delta[1]+delta[2]))/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = -delta[0]*(delta[1]+delta[2])/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = -delta[0]*delta[1]/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[1] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = delta[1]*delta[2]/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = -(delta[0]+delta[1])*delta[2]/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = (delta[2]*(delta[0]+2.*delta[1])-delta[1]*(delta[0]+delta[1]))/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = delta[1]*(delta[0]+delta[1])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[2] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = -delta[2]*(delta[1]+delta[2])/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = sumDelta*delta[2]/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = -sumDelta*(delta[1]+delta[2])/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = ((3.*sumDelta-delta[0])*delta[2]+sumDelta*delta[1])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[3] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        return(out);
    }

    template<class T,class T2>
    static std::vector<T> fourPointSecondDerivative(const std::vector<T> &data, const std::vector<T2> &base)
    {
        if(data.size()!=base.size())MYTHROW("Problems in utilities::fourPointSecondDerivative(): Inconsistent input!\n");
        if(data.size()!=4)MYTHROW("Problems in utilities::fourPointSecondDerivative(): Input not of size 4!\n");

        std::vector<T> out(data.size(),0.),
                  coeffs(data.size(),0.);

        std::vector<T2> delta(0);

        for(size_t point(0); point < data.size()-1; ++point) delta.push_back(base[point+1]-base[point]);

        T2 sumDelta(base.end()-base.begin());

        coeffs[0] =  2.*(3.*delta[0]+2.*delta[1]+delta[2])/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = -2.*(2.*(delta[0]+delta[1])+delta[2])/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] =  2.*(delta[0]+sumDelta)/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = -2.*(2.*delta[1]+delta[2])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[0] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = 2.*(2.*delta[1]+delta[2])/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = 2.*(delta[0]-2.*delta[1]-delta[2])/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = 2.*(sumDelta-2.*delta[0])/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = 2.*(delta[0]-delta[1])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[1] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = 2.*(delta[2]-delta[1])/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] = 2.*(sumDelta-2.*delta[2])/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = 2.*(delta[2]-2.*delta[1]-delta[0])/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] = 2.*(delta[0]+2.*delta[1])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[2] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = -2.*(delta[1]+2.*delta[2])/(delta[0]*sumDelta*(delta[0]+delta[1]));
        coeffs[1] =  2.*(sumDelta+delta[2])/(delta[0]*delta[1]*(delta[0]+delta[2]));
        coeffs[2] = -2.*(2.*sumDelta-delta[0])/(delta[1]*delta[2]*(delta[0]+delta[1]));
        coeffs[3] =  2.*(delta[0]+2.*delta[1]+3.*delta[2])/(sumDelta*delta[2]*(delta[1]+delta[2]));

        out[3] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];


        return(out);
    }

    //// NOTE: For the five point calculations, the reference does not give the values for the intermediate points.
    ////       I have computed the first derivative at the second point but I don't feel like computing the rest..
    template<class T,class T2>
    static std::vector<T> fivePointFirstDerivative(const std::vector<T> &data, const std::vector<T2> &base)
    {

        if(data.size()!=base.size())MYTHROW("Problems in utilities::fivePointFirstDerivative(): Inconsistent input!\n");
        if(data.size()!=5)MYTHROW("Problems in utilities::fivePointFirstDerivative(): Input not of size 5!\n");

        std::vector<T> out(data.size(),0.),
                  coeffs(data.size(),0.);

        std::vector<T2> delta(0);

        for(size_t point(0); point < data.size()-1; ++point) delta.push_back(base[point+1]-base[point]);

        std::vector<T2> sumDelta(2);
        sumDelta[0]=delta[0]+delta[1]+delta[2];
        sumDelta[1]=base.end()-base.begin();

        coeffs[0] = (2.*delta[0]+delta[1])*sumDelta[0]*sumDelta[1]/(sumDelta[0]*sumDelta[1]*delta[0]*(delta[0]+delta[1]));
        coeffs[1] = sumDelta[0]*sumDelta[1]*(delta[0]+delta[1])/(delta[0]*delta[1]*(delta[0]+delta[2])*(sumDelta[1]-delta[0]));
        coeffs[2] = -delta[0]*sumDelta[0]*sumDelta[1]/((delta[0]+delta[1])*delta[1]*delta[2]*(delta[2]+delta[3]));
        coeffs[3] = delta[0]*(delta[0]+delta[1])*sumDelta[1]/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] = delta[0]*sumDelta[0]*(delta[0]+delta[1])/(sumDelta[1]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])*delta[3]);

        out[0] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        coeffs[0] =  delta[1]*(delta[1]+delta[2])*(delta[0]-sumDelta[1])/(sumDelta[0]*sumDelta[1]*delta[0]*(delta[0]+delta[1]));
        coeffs[1] = -(delta[0]*delta[1]*(2.*delta[1]+2.*delta[2]+delta[3])-(delta[1]+delta[2])*(delta[0]-delta[1])*(delta[0]-sumDelta[1]))/(delta[0]*delta[1]*(delta[0]+delta[2])*(sumDelta[1]-delta[0]));
        coeffs[2] = delta[0]*(delta[1]+delta[2])*(sumDelta[1]-delta[0])/((delta[0]+delta[1])*delta[1]*delta[2]*(delta[2]+delta[3]));
        coeffs[3] = delta[0]*delta[1]*(delta[0]-sumDelta[1])/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] = delta[0]*delta[1]*(delta[2]+delta[3])/(sumDelta[1]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])*delta[3]);

        out[1] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        coeffs[0] = delta[1]*delta[2]*(delta[2]+delta[3])/(sumDelta[0]*sumDelta[1]*delta[0]*(delta[0]+delta[1]));
        coeffs[1] = -(delta[0]+delta[1])*delta[2]*(delta[2]+delta[3])/(delta[0]*delta[1]*(delta[0]+delta[2])*(sumDelta[1]-delta[0]));
        coeffs[2] = ((delta[0]+2.*delta[2])*(delta[2]+delta[3])*delta[2]-(delta[0]+delta[1])*(2.*delta[2]+delta[3])*delta[1])/((delta[0]+delta[1])*delta[1]*delta[2]*(delta[2]+delta[3]));
        coeffs[3] = delta[1]*(delta[0]+delta[1])*(delta[2]+delta[3])/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] = -delta[0]*sumDelta[0]*(delta[0]+delta[1])/(sumDelta[1]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])*delta[3]);

        out[2] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        //   coeffs[0] = /(sumDelta[0]*sumDelta[1]*delta[0]*(delta[0]+delta[1]));
        //   coeffs[1] = /(delta[0]*delta[1]*(delta[0]+delta[2])*(sumDelta[1]-delta[0]));
        //   coeffs[2] = /((delta[0]+delta[1])*delta[1]*delta[2]*(delta[2]+delta[3]));
        //   coeffs[3] = /(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        //   coeffs[4] = /(sumDelta[1]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])*delta[3]);
        //
        //   out[3] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        coeffs[0] = delta[3]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])/(sumDelta[0]*sumDelta[1]*delta[0]*(delta[0]+delta[1]));
        coeffs[1] = -sumDelta[1]*delta[3]*(delta[2]+delta[3])/(delta[0]*delta[1]*(delta[0]+delta[2])*(sumDelta[1]-delta[0]));
        coeffs[2] = sumDelta[1]*(sumDelta[1]-delta[0])*delta[3]/((delta[0]+delta[1])*delta[1]*delta[2]*(delta[2]+delta[3]));
        coeffs[3] = -sumDelta[1]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] = ((2.*sumDelta[1]-delta[0])*(delta[2]+delta[3])*delta[3]+(sumDelta[1]-delta[0])*(delta[2]+2.*delta[3])*sumDelta[1])/(sumDelta[1]*(sumDelta[1]-delta[0])*(delta[2]+delta[3])*delta[3]);

        out[4] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        return(out);
    }

    template<class T,class T2>
    static std::vector<T> fivePointSecondDerivative(const std::vector<T> &data, const std::vector<T2> &base)
    {

        if(data.size()!=base.size())MYTHROW("Problems in utilities::fivePointSecondDerivative(): Inconsistent input!\n");
        if(data.size()!=5)MYTHROW("Problems in utilities::fivePointSecondDerivative(): Input not of size 5!\n");

        std::vector<T> out(data.size(),0.),
                  coeffs(data.size(),0.);

        std::vector<T2> delta(0);

        for(size_t point(0); point < data.size()-1; ++point) delta.push_back(base[point+1]-base[point]);

        std::vector<T2> sumDelta(2);
        sumDelta[0]=delta[0]+delta[1]+delta[2];
        sumDelta[1]=base.end()-base.begin();

        coeffs[0] = ((3.*delta[0]+2.*delta[1])*(2.*sumDelta[1]-delta[3])+2.*sumDelta[0]*sumDelta[1]+delta[0]*(4.*sumDelta[1]-2.*delta[2]-3.*delta[3]))/(delta[0]*sumDelta[0]*sumDelta[1]*(delta[0]+delta[1]));
        coeffs[1] = -2.*((delta[0]+delta[1])*(2.*sumDelta[2]-delta[3])+sumDelta[0]*sumDelta[1])/(delta[0]*delta[1]*(delta[0]+delta[2])*(delta[1]+delta[2]+delta[3]));
        coeffs[2] =  2.*(delta[0]*(2.*sumDelta[1]-delta[3])+sumDelta[0]*sumDelta[1])/(delta[1]*delta[2]*(delta[0]+delta[1])*(delta[2]+delta[3]));
        coeffs[3] = -2.*(delta[0]*(delta[2]+delta[1]+sumDelta[1])+sumDelta[1]*(delta[0]+delta[1]))/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] =  2.*(delta[0]*(2.*sumDelta[1]-delta[2])+sumDelta[1]*(delta[0]+delta[1]))/(sumDelta[1]*(delta[1]+delta[2]+delta[3])*(delta[2]+delta[3])*delta[3]);

        out[0] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        //   coeffs[0] = /();
        //   coeffs[1] = /();
        //   coeffs[2] = /();
        //   coeffs[3] = /();
        //
        //
        //   out[1] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] = -2.*(delta[1]*(2.*delta[2]+delta[3])+delta[2]*(delta[2]+delta[3]))/(delta[0]*sumDelta[0]*sumDelta[1]*(delta[0]+delta[1]));
        coeffs[1] =  2.*((delta[1]+delta[2])*(2.*delta[2]+delta[3])-delta[2]*(delta[2]+delta[3]))/(delta[0]*delta[1]*(delta[0]+delta[2])*(delta[1]+delta[2]+delta[3]));
        coeffs[2] =  2.*(delta[1]*(delta[0]+delta[1])-(delta[0]+2.*delta[1])*(2.*delta[2]+delta[3])+delta[2]*(delta[2]+delta[3]))/(delta[1]*delta[2]*(delta[0]+delta[1])*(delta[2]+delta[3]));
        coeffs[3] =  2.*((delta[0]+2.*delta[1])*(delta[2]+delta[3])-delta[1]*(delta[0]+delta[1]))/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] =  2.*(delta[1]*(delta[0]+delta[1])-delta[2]*(delta[0]+2.*delta[1]))/(sumDelta[1]*(delta[1]+delta[2]+delta[3])*(delta[2]+delta[3])*delta[3]);

        out[2] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];

        //   coeffs[0] = /();
        //   coeffs[1] = /();
        //   coeffs[2] = /();
        //   coeffs[3] = /();
        //
        //   out[3] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3];

        coeffs[0] =  (delta[3]*(delta[1]+3.*(delta[2]+delta[3]))+(sumDelta[1]-delta[0])*(2.*delta[2]+3.*delta[3]))/(delta[0]*sumDelta[0]*sumDelta[1]*(delta[0]+delta[1]));
        coeffs[1] = -(delta[3]*(sumDelta[1]+delta[2]+delta[3])+sumDelta[1]*(2.*delta[2]+3.*delta[3])+delta[3]*(delta[2]+delta[3]))/(delta[0]*delta[1]*(delta[0]+delta[2])*(delta[1]+delta[2]+delta[3]));
        coeffs[2] =  2.*(delta[3]*(2.*sumDelta[1]-delta[0])+sumDelta[1]*(sumDelta[1]-delta[0]))/(delta[1]*delta[2]*(delta[0]+delta[1])*(delta[2]+delta[3]));
        coeffs[3] = -2.*((2.*sumDelta[1]-delta[0])*(delta[2]+delta[3])+sumDelta[1]*(sumDelta[1]-delta[0]))/(sumDelta[0]*(delta[1]+delta[2])*delta[2]*delta[3]);
        coeffs[4] =  2.*((2.*sumDelta[1]-delta[0])*(delta[2]+2.*delta[3])+delta[3]*(delta[2]+delta[3])+sumDelta[1]*(sumDelta[1]-delta[0]))/(sumDelta[1]*(delta[1]+delta[2]+delta[3])*(delta[2]+delta[3])*delta[3]);

        out[3] = coeffs[0]*data[0]+coeffs[1]*data[1]+coeffs[2]*data[2]+coeffs[3]*data[3]+coeffs[4]*data[4];


        return(out);
    }



    /*
 * Pop_front for vectors
 */
    template<typename T>
    static void pop_front(std::vector<T>& vec)
    {
        assert(!vec.empty());
        vec.erase(vec.begin());
    }

    /*
   Function to invert a 2x2 matrix.
*/
    template<typename T>
    inline static void matinv2x2(std::vector<T> &a)
    {
        T det = Det2x2(a);

        T eps = 1.e-35;

        if(std::abs(det) < eps)
        {
            if(det <  0.) det = -eps;
            if(det >= 0.) det = eps;
        }

        std::vector<T> ainv(a);

        ainv[2*0+0] /= det;
        ainv[2*0+1] = -a[2*1+0]/det;
        ainv[2*1+0] = -a[2*0+1]/det;
        ainv[2*1+1] /= det;

        a = ainv;
    }

    /*
   Function to invert a 3x3 matrix.
*/
    template<typename T>
    inline static void matinv3x3(std::vector<T> &a)
    {
        T det = Det3x3(a);

        T eps = 1.e-35;

        if(std::abs(det) < eps)
        {
            if(det <  0.) det = -eps;
            if(det >= 0.) det = eps;
        }

        std::vector<T> ainvt(9,0.),
                       menor(4,0.);

        menor[2*0+0] = a[3*1+1];
        menor[2*0+1] = a[3*1+2];
        menor[2*1+0] = a[3*2+1];
        menor[2*1+1] = a[3*2+2];
        ainvt[3*0+0] = Det2x2(menor)/det;

        menor[2*0+0] = a[3*1+0];
        menor[2*0+1] = a[3*1+2];
        menor[2*1+0] = a[3*2+0];
        menor[2*1+1] = a[3*2+2];
        ainvt[3*0+1] = -Det2x2(menor)/det;


        menor[2*0+0] = a[3*1+0];
        menor[2*0+1] = a[3*1+1];
        menor[2*1+0] = a[3*2+0];
        menor[2*1+1] = a[3*2+1];
        ainvt[3*0+2] = Det2x2(menor)/det;

        menor[2*0+0] = a[3*0+1];
        menor[2*0+1] = a[3*0+2];
        menor[2*1+0] = a[3*2+1];
        menor[2*1+1] = a[3*2+2];
        ainvt[3*1+0] = -Det2x2(menor)/det;


        menor[2*0+0] = a[3*0+0];
        menor[2*0+1] = a[3*0+2];
        menor[2*1+0] = a[3*2+0];
        menor[2*1+1] = a[3*2+2];
        ainvt[3*1+1] = Det2x2(menor)/det;

        menor[2*0+0] = a[3*0+0];
        menor[2*0+1] = a[3*0+1];
        menor[2*1+0] = a[3*2+0];
        menor[2*1+1] = a[3*2+1];
        ainvt[3*1+2] = -Det2x2(menor)/det;

        menor[2*0+0] = a[3*0+1];
        menor[2*0+1] = a[3*0+2];
        menor[2*1+0] = a[3*1+1];
        menor[2*1+1] = a[3*1+2];
        ainvt[3*2+0] = Det2x2(menor)/det;

        menor[2*0+0] = a[3*0+0];
        menor[2*0+1] = a[3*0+2];
        menor[2*1+0] = a[3*1+0];
        menor[2*1+1] = a[3*1+2];
        ainvt[3*2+1] = -Det2x2(menor)/det;

        menor[2*0+0] = a[3*0+0];
        menor[2*0+1] = a[3*0+1];
        menor[2*1+0] = a[3*1+0];
        menor[2*1+1] = a[3*1+1];
        ainvt[3*2+2] = Det2x2(menor)/det;


        // Transpose
        for(size_t i(0); i < 3; ++i)
        {
            for(size_t j(0); j < 3; ++j)
            {
                a[3*i+j] = ainvt[3*j+i];
            }
        }
    }
    /*
   Function to invert a 5x5 matrix.
*/
    template<typename T>
    inline static void matinv5x5(std::vector<T> &a)
    {
        T piv(0.);

        piv	   = 1. / a[4*4+4];
        a[4*4  ] = piv*a[4*4  ];
        a[4*4+1] = piv*a[4*4+1];
        a[4*4+2] = piv*a[4*4+2];
        a[4*4+3] = piv*a[4*4+3];
        a[4*4+4] = piv;

        piv	   = a[4*3+4];
        a[4*3  ] = a[4*3  ] - piv*a[4*4  ];
        a[4*3+1] = a[4*3+1] - piv*a[4*4+1];
        a[4*3+2] = a[4*3+2] - piv*a[4*4+2];
        a[4*3+3] = a[4*3+3] - piv*a[4*4+3];
        a[4*3+4] =	      - piv*a[4*4+4];

        piv	   = a[4*2+4];
        a[4*2  ] = a[4*2  ] - piv*a[4*4  ];
        a[4*2+1] = a[4*2+1] - piv*a[4*4+1];
        a[4*2+2] = a[4*2+2] - piv*a[4*4+2];
        a[4*2+3] = a[4*2+3] - piv*a[4*4+3];
        a[4*2+4] =	      - piv*a[4*4+4];

        piv	   = a[4*1+4];
        a[4*1  ] = a[4*1  ] - piv*a[4*4  ];
        a[4*1+1] = a[4*1+1] - piv*a[4*4+1];
        a[4*1+2] = a[4*1+2] - piv*a[4*4+2];
        a[4*1+3] = a[4*1+3] - piv*a[4*4+3];
        a[4*1+4] =	      - piv*a[4*4+4];

        piv	   = a[4*0+4];
        a[4*0  ] = a[4*0  ] - piv*a[4*4  ];
        a[4*0+1] = a[4*0+1] - piv*a[4*4+1];
        a[4*0+2] = a[4*0+2] - piv*a[4*4+2];
        a[4*0+3] = a[4*0+3] - piv*a[4*4+3];
        a[4*0+4] =	      - piv*a[4*4+4];

        piv	   = 1. / a[4*3+3];
        a[4*3  ] = piv*a[4*3  ];
        a[4*3+1] = piv*a[4*3+1];
        a[4*3+2] = piv*a[4*3+2];
        a[4*3+3] = piv;
        a[4*3+4] = piv*a[4*3+4];

        piv	   = a[4*4+3];
        a[4*4  ] = a[4*4  ] - piv*a[4*3  ];
        a[4*4+1] = a[4*4+1] - piv*a[4*3+1];
        a[4*4+2] = a[4*4+2] - piv*a[4*3+2];
        a[4*4+3] =	      - piv*a[4*3+3];
        a[4*4+4] = a[4*4+4] - piv*a[4*3+4];

        piv	   = a[4*2+3];
        a[4*2  ] = a[4*2  ] - piv*a[4*3  ];
        a[4*2+1] = a[4*2+1] - piv*a[4*3+1];
        a[4*2+2] = a[4*2+2] - piv*a[4*3+2];
        a[4*2+3] =	      - piv*a[4*3+3];
        a[4*2+4] = a[4*2+4] - piv*a[4*3+4];

        piv	   = a[4*1+3];
        a[4*1  ] = a[4*1  ] - piv*a[4*3  ];
        a[4*1+1] = a[4*1+1] - piv*a[4*3+1];
        a[4*1+2] = a[4*1+2] - piv*a[4*3+2];
        a[4*1+3] =	      - piv*a[4*3+3];
        a[4*1+4] = a[4*1+4] - piv*a[4*3+4];

        piv	   = a[4*0+3];
        a[4*0  ] = a[4*0  ] - piv*a[4*3  ];
        a[4*0+1] = a[4*0+1] - piv*a[4*3+1];
        a[4*0+2] = a[4*0+2] - piv*a[4*3+2];
        a[4*0+3] =	      - piv*a[4*3+3];
        a[4*0+4] = a[4*0+4] - piv*a[4*3+4];

        piv	   = 1. / a[4*2+2];
        a[4*2  ] = piv*a[4*2  ];
        a[4*2+1] = piv*a[4*2+1];
        a[4*2+2] = piv;
        a[4*2+3] = piv*a[4*2+3];
        a[4*2+4] = piv*a[4*2+4];

        piv      = a[4*4+2];
        a[4*4  ] = a[4*4  ] - piv*a[4*2  ];
        a[4*4+1] = a[4*4+1] - piv*a[4*2+1];
        a[4*4+2] =	      - piv*a[4*2+2];
        a[4*4+3] = a[4*4+3] - piv*a[4*2+3];
        a[4*4+4] = a[4*4+4] - piv*a[4*2+4];

        piv	   = a[4*3+2];
        a[4*3  ] = a[4*3  ] - piv*a[4*2  ];
        a[4*3+1] = a[4*3+1] - piv*a[4*2+1];
        a[4*3+2] =	      - piv*a[4*2+2];
        a[4*3+3] = a[4*3+3] - piv*a[4*2+3];
        a[4*3+4] = a[4*3+4] - piv*a[4*2+4];

        piv	 = a[4*1+2];
        a[4*1  ] = a[4*1  ] - piv*a[4*2  ];
        a[4*1+1] = a[4*1+1] - piv*a[4*2+1];
        a[4*1+2] =	      - piv*a[4*2+2];
        a[4*1+3] = a[4*1+3] - piv*a[4*2+3];
        a[4*1+4] = a[4*1+4] - piv*a[4*2+4];

        piv	   = a[4*0+2];
        a[4*0  ] = a[4*0  ] - piv*a[4*2  ];
        a[4*0+1] = a[4*0+1] - piv*a[4*2+1];
        a[4*0+2] =	      - piv*a[4*2+2];
        a[4*0+3] = a[4*0+3] - piv*a[4*2+3];
        a[4*0+4] = a[4*0+4] - piv*a[4*2+4];

        piv	   = 1. / a[4*1+1];
        a[4*1  ] = piv*a[4*1  ];
        a[4*1+1] = piv;
        a[4*1+2] = piv*a[4*1+2];
        a[4*1+3] = piv*a[4*1+3];
        a[4*1+4] = piv*a[4*1+4];

        piv	   = a[4*4+1];
        a[4*4  ] = a[4*4  ] - piv*a[4*1  ];
        a[4*4+1] =	      - piv*a[4*1+1];
        a[4*4+2] = a[4*4+2] - piv*a[4*1+2];
        a[4*4+3] = a[4*4+3] - piv*a[4*1+3];
        a[4*4+4] = a[4*4+4] - piv*a[4*1+4];

        piv      = a[4*3+1];
        a[4*3  ] = a[4*3  ] - piv*a[4*1  ];
        a[4*3+1] =	      - piv*a[4*1+1];
        a[4*3+2] = a[4*3+2] - piv*a[4*1+2];
        a[4*3+3] = a[4*3+3] - piv*a[4*1+3];
        a[4*3+4] = a[4*3+4] - piv*a[4*1+4];

        piv	   = a[4*2+1];
        a[4*2  ] = a[4*2  ] - piv*a[4*1  ];
        a[4*2+1] =	      - piv*a[4*1+1];
        a[4*2+2] = a[4*2+2] - piv*a[4*1+2];
        a[4*2+3] = a[4*2+3] - piv*a[4*1+3];
        a[4*2+4] = a[4*2+4] - piv*a[4*1+4];

        piv	   = a[4*0+1];
        a[4*0  ] = a[4*0  ] - piv*a[4*1  ];
        a[4*0+1] =	      - piv*a[4*1+1];
        a[4*0+2] = a[4*0+2] - piv*a[4*1+2];
        a[4*0+3] = a[4*0+3] - piv*a[4*1+3];
        a[4*0+4] = a[4*0+4] - piv*a[4*1+4];

        piv	   = 1. / a[4*0  ];
        a[4*0  ] = piv;
        a[4*0+1] = piv*a[4*0+1];
        a[4*0+2] = piv*a[4*0+2];
        a[4*0+3] = piv*a[4*0+3];
        a[4*0+4] = piv*a[4*0+4];

        piv	   = a[4*4  ];
        a[4*4  ] =	      - piv*a[4*0  ];
        a[4*4+1] = a[4*4+1] - piv*a[4*0+1];
        a[4*4+2] = a[4*4+2] - piv*a[4*0+2];
        a[4*4+3] = a[4*4+3] - piv*a[4*0+3];
        a[4*4+4] = a[4*4+4] - piv*a[4*0+4];

        piv	   = a[4*3  ];
        a[4*3  ] =	      - piv*a[4*0  ];
        a[4*3+1] = a[4*3+1] - piv*a[4*0+1];
        a[4*3+2] = a[4*3+2] - piv*a[4*0+2];
        a[4*3+3] = a[4*3+3] - piv*a[4*0+3];
        a[4*3+4] = a[4*3+4] - piv*a[4*0+4];

        piv	   = a[4*2  ];
        a[4*2  ] =	      - piv*a[4*0  ];
        a[4*2+1] = a[4*2+1] - piv*a[4*0+1];
        a[4*2+2] = a[4*2+2] - piv*a[4*0+2];
        a[4*2+3] = a[4*2+3] - piv*a[4*0+3];
        a[4*2+4] = a[4*2+4] - piv*a[4*0+4];

        piv	   = a[4*1  ];
        a[4*1  ] =          - piv*a[4*0  ];
        a[4*1+1] = a[4*1+1] - piv*a[4*0+1];
        a[4*1+2] = a[4*1+2] - piv*a[4*0+2];
        a[4*1+3] = a[4*1+3] - piv*a[4*0+3];
        a[4*1+4] = a[4*1+4] - piv*a[4*0+4];
    }

    template<typename T>
    inline static T Det2x2(const std::vector<T> &a)
    {
        T det = a[2*0+0] * a[2*1+1] -
                a[2*0+1] * a[2*1+0] ;
        return det;
    }

    template<typename T>
    inline static T Det3x3(const std::vector<T> &a)
    {
        T det = a[3*0+0] * a[3*1+1] * a[3*2+2] +
                a[3*0+1] * a[3*2+0] * a[3*1+2] +
                a[3*0+2] * a[3*1+0] * a[3*2+1] -
                a[3*0+2] * a[3*1+1] * a[3*2+0] -
                a[3*0+0] * a[3*2+1] * a[3*1+2] -
                a[3*2+2] * a[3*1+0] * a[3*0+1];
        return det;
    }

    template<typename T>
    inline static T Det4x4(const std::vector<T> &a)
    {
        std::vector<T> Menor(9,0.);

        Menor[3*0+0] = a[4*1+1];
        Menor[3*0+1] = a[4*1+2];
        Menor[3*0+2] = a[4*1+3];

        Menor[3*1+0] = a[4*2+1];
        Menor[3*1+1] = a[4*2+2];
        Menor[3*1+2] = a[4*2+3];

        Menor[3*2+0] = a[4*3+1];
        Menor[3*2+1] = a[4*3+2];
        Menor[3*2+2] = a[4*3+3];

        T det1 = Det3x3(Menor);

        Menor[3*0+0] = a[4*1+0];
        Menor[3*1+0] = a[4*2+0];
        Menor[3*2+0] = a[4*3+0];

        T det2 = Det3x3(Menor);

        Menor[3*0+1] = a[4*1+1];
        Menor[3*1+1] = a[4*2+1];
        Menor[3*2+1] = a[4*3+1];

        T det3 = Det3x3(Menor);

        Menor[3*0+2] = a[4*1+2];
        Menor[3*1+2] = a[4*2+2];
        Menor[3*2+2] = a[4*3+2];

        T det4 = Det3x3(Menor);

        T det = a[4*0+0] * det1 - a[4*0+1] * det2 + a[4*0+2] * det3 - a[4*0+3] * det4;

        return det;
    }

    /*
   Function to invert a 4x4 matrix.
*/
    template<typename T>
    inline static void matinv4x4(std::vector<T> &m)
    {

        std::vector<T> inv(16,0.);
        T det(0.);

        inv[0] = m[5]  * m[10] * m[15] -
                m[5]  * m[11] * m[14] -
                m[9]  * m[6]  * m[15] +
                m[9]  * m[7]  * m[14] +
                m[13] * m[6]  * m[11] -
                m[13] * m[7]  * m[10];

        inv[4] = -m[4]  * m[10] * m[15] +
                m[4]  * m[11] * m[14] +
                m[8]  * m[6]  * m[15] -
                m[8]  * m[7]  * m[14] -
                m[12] * m[6]  * m[11] +
                m[12] * m[7]  * m[10];

        inv[8] = m[4]  * m[9] * m[15] -
                m[4]  * m[11] * m[13] -
                m[8]  * m[5] * m[15] +
                m[8]  * m[7] * m[13] +
                m[12] * m[5] * m[11] -
                m[12] * m[7] * m[9];

        inv[12] = -m[4]  * m[9] * m[14] +
                m[4]  * m[10] * m[13] +
                m[8]  * m[5] * m[14] -
                m[8]  * m[6] * m[13] -
                m[12] * m[5] * m[10] +
                m[12] * m[6] * m[9];

        inv[1] = -m[1]  * m[10] * m[15] +
                m[1]  * m[11] * m[14] +
                m[9]  * m[2] * m[15] -
                m[9]  * m[3] * m[14] -
                m[13] * m[2] * m[11] +
                m[13] * m[3] * m[10];

        inv[5] = m[0]  * m[10] * m[15] -
                m[0]  * m[11] * m[14] -
                m[8]  * m[2] * m[15] +
                m[8]  * m[3] * m[14] +
                m[12] * m[2] * m[11] -
                m[12] * m[3] * m[10];

        inv[9] = -m[0]  * m[9] * m[15] +
                m[0]  * m[11] * m[13] +
                m[8]  * m[1] * m[15] -
                m[8]  * m[3] * m[13] -
                m[12] * m[1] * m[11] +
                m[12] * m[3] * m[9];

        inv[13] = m[0]  * m[9] * m[14] -
                m[0]  * m[10] * m[13] -
                m[8]  * m[1] * m[14] +
                m[8]  * m[2] * m[13] +
                m[12] * m[1] * m[10] -
                m[12] * m[2] * m[9];

        inv[2] = m[1]  * m[6] * m[15] -
                m[1]  * m[7] * m[14] -
                m[5]  * m[2] * m[15] +
                m[5]  * m[3] * m[14] +
                m[13] * m[2] * m[7] -
                m[13] * m[3] * m[6];

        inv[6] = -m[0]  * m[6] * m[15] +
                m[0]  * m[7] * m[14] +
                m[4]  * m[2] * m[15] -
                m[4]  * m[3] * m[14] -
                m[12] * m[2] * m[7] +
                m[12] * m[3] * m[6];

        inv[10] = m[0]  * m[5] * m[15] -
                m[0]  * m[7] * m[13] -
                m[4]  * m[1] * m[15] +
                m[4]  * m[3] * m[13] +
                m[12] * m[1] * m[7] -
                m[12] * m[3] * m[5];

        inv[14] = -m[0]  * m[5] * m[14] +
                m[0]  * m[6] * m[13] +
                m[4]  * m[1] * m[14] -
                m[4]  * m[2] * m[13] -
                m[12] * m[1] * m[6] +
                m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] +
                m[1] * m[7] * m[10] +
                m[5] * m[2] * m[11] -
                m[5] * m[3] * m[10] -
                m[9] * m[2] * m[7] +
                m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] -
                m[0] * m[7] * m[10] -
                m[4] * m[2] * m[11] +
                m[4] * m[3] * m[10] +
                m[8] * m[2] * m[7] -
                m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] +
                m[0] * m[7] * m[9] +
                m[4] * m[1] * m[11] -
                m[4] * m[3] * m[9] -
                m[8] * m[1] * m[7] +
                m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] -
                m[0] * m[6] * m[9] -
                m[4] * m[1] * m[10] +
                m[4] * m[2] * m[9] +
                m[8] * m[1] * m[6] -
                m[8] * m[2] * m[5];

        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        T eps = 1.e-35;
        if (det < eps && det >= 0.)
        {
            det = eps;
        }
        else if (det < 0. && det > -eps)
        {
            det = -eps;
        }

        for (int i(0); i<16; ++i)
        {
            m[i] = inv[i]/det;
        }


        return;
    }

    /*
   Remove repeated elements in a vector
*/
    template<typename T>
    inline static void FilterDuplicates(std::vector<T> &list)
    {
        std::sort(list.begin(),list.end());

        list.erase( unique(list.begin(),list.end() ), list.end() );
    }

    /*
   Remove elements in a vector if there are repeated indices in a corresponding list
*/
    template<typename T>
    inline static void FilterDuplicates(std::vector<size_t> &list,std::vector<T> &data)
    {



        if(list.size() != data.size()) MYTHROW("Problems in utilities::FilterDuplicates(list,data): Inconsistent input vector sizes!");

        std::vector<size_t> sortedList(0),
                idxList(0),
                tmpList(0);

        std::vector<T> tmpData(0);

        sort_index(list,sortedList,idxList);


        int idx(0);


        for(size_t i(0); i < list.size(); ++i)
        {
            idx = idxList[i];

            if(idx == 0)
            {
                tmpList.push_back(list[i]);
                tmpData.push_back(data[i]);
            }
            else
            {
                if(sortedList[idx] > sortedList[idx - 1])
                {
                    tmpList.push_back(list[i]);
                    tmpData.push_back(data[i]);
                }
            }

        }

        list.clear();
        data.clear();

        list = tmpList;
        data = tmpData;


    }

    /*
    Find the indexes of occurrences of "findFor" in "theArray"
 */
    template<typename T>
    static std::vector<size_t> searchList(const std::vector<T> &theArray,const T &findFor)
    {
        std::vector<size_t> indexes(0);
        for (size_t i = 0; i < theArray.size(); ++i)
        {
            if (std::abs(theArray[i] - findFor)<1.e3*std::numeric_limits<T>::epsilon())
            {
                indexes.push_back(i);
            }
        }
        return(indexes);
    }

    /*
 *
 *  Filter high frequency noise by interpolating into a coarser mesh, and back to the original mesh.
 *
 *   Input:
 *       data: The discrete curve points.
 *       base: The discrete base mesh.
 *       skip: Number of samples to skip.
 *       method:
 *              -l: Linear interpolations
 *              -c: Cubic interpolations
 *              -s: Cubic interpolation to coarse, linear back to fine
 *              -n: Linear interpolation to coarse, cubic nack to fine
 */
    template<class T>
    static std::vector<T> CutreFilter(const std::vector<T> &data,
                                 const std::vector<T> &base,
                                 const size_t &skip,
                                 const char &method='c')
    {
        std::vector<T> filteredSignal(0),
                  reducedData(0),
                  xNew(0);

        xNew.push_back(base[0]);
        for(size_t p(1); p < data.size()-1; ++p)
        {
            if( (p % skip) == 0)
            {
                xNew.push_back(base[p]);
            }
        }
        xNew.push_back(base[data.size()-1]);

        switch (method) {
        case 'l':
        {
            reducedData = interpolate(base,data,xNew,'l');
            filteredSignal = interpolate(xNew,reducedData,base,'l');
            break;
        }
        case 'c':
        {
            reducedData = interpolate(base,data,xNew,'c');
            filteredSignal = interpolate(xNew,reducedData,base,'c');
            break;
        }
        case 'n':
        {
            reducedData = interpolate(base,data,xNew,'l');
            filteredSignal = interpolate(xNew,reducedData,base,'c');
            break;
        }
        case 's':
        {
            reducedData = interpolate(base,data,xNew,'c');
            filteredSignal = interpolate(xNew,reducedData,base,'l');
            break;
        }
        default:
        {
            reducedData = interpolate(base,data,xNew,'l');
            filteredSignal = interpolate(xNew,reducedData,base,'c');
            break;
        }
        }



        return(filteredSignal);

    }



    /*
 *
 *  Apply a polynomial least quares fit
 *
 *   Input:
 *       data: The discrete curve points.
 *       base: The discrete base mesh.
 *       order: Order of the polynomial filter.
 *
 *  Output: The coefficients of the polynomial
 *
 */
    template<class T>
    static std::vector<T> LeastSqFit(const std::vector<T> &data, const std::vector<T> &base, unsigned short order)
    {

        size_t N(data.size());

        if(N != base.size()) MYTHROW("Trouble in utilities::LeastSqFit(), inconsistent input!\n");

        if(N < order) MYTHROW("Trouble in utilities::LeastSqFit(), order of polynomial greater than stencil size!\n");
        if(N < 3) MYTHROW("Trouble in utilities::LeastSqFit(), no less than 3 points allowed!\n");


        Eigen::VectorXd source(order+1);

        Eigen::MatrixXd X(order+1,N);

        for(size_t row(0); row < order+1; ++row)
        {
            source(row) = data[row];

            for(size_t col(0); col < N; ++col)
            {
                X(row,col) = std::pow(base[col],row);
            }
        }

        Eigen::MatrixXd matrix = X*X.transpose();

        Eigen::VectorXd coeffsEig = matrix.inverse()*X*source;

        std::vector<T> coeffs(order+1,0.);
        for(size_t i(0); i < order+1; ++i) coeffs[i] = coeffsEig(i);

        return(coeffs);
    }

    /*
 *
 *  Filter high frequencies with a polynomial least squares fit.
 *
 *   Input:
 *       data: The discrete curve points.
 *       base: The discrete base mesh.
 *       stencil: Number of points to consider in the LSq fitting
 *       order: Order of the polynomial filter.
 *       der: Give the derivative when true
 *       isClosedCurve: Consider a closed and periodic curve if true
 *
 */
    template<class T>
    static std::vector<T> LeastSqFilter(const std::vector<T> &data,
                                   const std::vector<T> &base,
                                   const int &stencil,
                                   const int &order,
                                   bool der=false,
                                   bool isClosedCurve=false)
    {
        int Dsiz(data.size()),
            Bsiz(base.size());

        if(Dsiz != Bsiz) MYTHROW("Trouble in utilities::LeastSqFilter(), inconsistent input!\n");
        if(stencil == order) MYTHROW("Trouble in utilities::LeastSqFilter(), the same order of polynomial as stencil size, that is an interpolation!\n");

        std::vector<T> filteredSignal(base.size(),0.),
                  reducedData(stencil,0.),
                  reducedBase(stencil,0.),
                  coeffs(0);

        int counterIni(floor(stencil/2)),
            counterF(base.size()-counterIni);


        for(int p(0); p < Bsiz; ++p)
        {

            if( (p-counterIni) < 0)
            {
                for(int j(0); j < counterIni; ++j)
                {
                    if(isClosedCurve)
                    {
                        reducedBase[j] = base[counterF + j];
                        reducedData[j] = data[counterF + j];
                    }
                    else
                    {
                        reducedBase[j] = base[0];
                        reducedData[j] = data[0];
                    }
                }
                for(int j(counterIni); j < stencil; ++j)
                {
                    reducedBase[j] = base[p + j - counterIni];
                    reducedData[j] = data[p + j - counterIni];
                }
            }
            else if( (counterF - p) < 0)
            {
                for(int j(0); j < counterIni; ++j)
                {
                    reducedBase[j] = base[p + j - counterIni];
                    reducedData[j] = data[p + j - counterIni];
                }
                for(int j(counterIni); j < stencil; ++j)
                {
                    if(isClosedCurve)
                    {
                        reducedBase[j] = base[j - counterIni];
                        reducedData[j] = data[j - counterIni];
                    }
                    else
                    {
                        reducedBase[j] = base[base.size()-1];
                        reducedData[j] = data[base.size()-1];
                    }
                }
            }
            else
            {
                for(int j(0); j < stencil; ++j)
                {
                    reducedBase[j] = base[p + j - counterIni];
                    reducedData[j] = data[p + j - counterIni];
                }
            }

            coeffs = LeastSqFit(reducedData, reducedBase,order);
            filteredSignal[p] = PolyEval(coeffs,base[p],der);

        }

        return(filteredSignal);

    }

    /*
 *
 * Obtain the pseudoinverse of a non-square matrix.
 *
 */
    static Eigen::MatrixXd pinv(const Eigen::MatrixXd& A, double pinvtoler=1.e-12)
    {

        if(A.rows() < A.cols())
        {
            MYTHROW("Error in pinv: less rows than columns!");
        }

        Eigen::BDCSVD<Eigen::MatrixXd> svdOfA(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

        const Eigen::MatrixXd U = svdOfA.matrixU();
        const Eigen::MatrixXd V = svdOfA.matrixV();


        Eigen::VectorXd singularValues_inv = svdOfA.singularValues();


        for( long i=0; i < A.cols(); i++)
        {
            if ( std::abs(svdOfA.singularValues()(i)) < pinvtoler )
            {
                singularValues_inv(i) = 0.;
            }
            else
            {
                singularValues_inv(i) = 1./svdOfA.singularValues()(i);
            }
        }


        const Eigen::MatrixXd mAdjointU = U.adjoint().block(0,0,singularValues_inv.rows(),U.adjoint().cols());

        return(V*singularValues_inv.asDiagonal()*mAdjointU);
    }

    // Convert from Eigen::Vector to std::vector, naive implementation
    static std::vector<double> eigenToStd_d(const Eigen::VectorXd &eigV)
    {
        std::vector<double> eigStd(0);
        for(unsigned int idx(0); idx<eigV.size(); ++idx) eigStd.push_back(eigV(idx));
        return(eigStd);

    }
    static std::vector<float> eigenToStd_f(const Eigen::VectorXf &eigV)
    {
        std::vector<float> eigStd(0);
        for(unsigned int idx(0); idx<eigV.size(); ++idx) eigStd.push_back(eigV(idx));
        return(eigStd);
    }
    static std::vector<int> eigenToStd_i(const Eigen::VectorXi &eigV)
    {
        std::vector<int> eigStd(0);
        for(unsigned int idx(0); idx<eigV.size(); ++idx) eigStd.push_back(eigV(idx));
        return(eigStd);
    }

    // Convert from Eigen::Matrix to std::vector<vector>
    static std::vector<std::vector<double>> eigenToStd_Md(const Eigen::MatrixXd &eigM)
    {
        unsigned int Nrows(eigM.rows());
        unsigned int Ncols(eigM.cols());

        std::vector<double> eigStd(Ncols);
        std::vector<std::vector<double>> MStd(Nrows,eigStd);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                MStd[row][col]=eigM(row,col);
            }
        }

        return(MStd);
    }

    static std::vector<std::vector<float>> eigenToStd_Mf(const Eigen::MatrixXf &eigM)
    {
        unsigned int Nrows(eigM.rows());
        unsigned int Ncols(eigM.cols());

        std::vector<float> eigStd(Ncols);
        std::vector<std::vector<float>> MStd(Nrows,eigStd);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                MStd[row][col]=eigM(row,col);
            }
        }

        return(MStd);
    }

    static std::vector<std::vector<int>> eigenToStd_Mi(const Eigen::MatrixXi &eigM)
    {
        unsigned int Nrows(eigM.rows());
        unsigned int Ncols(eigM.cols());

        std::vector<int> eigStd(Ncols);
        std::vector<std::vector<int>> MStd(Nrows,eigStd);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                MStd[row][col]=eigM(row,col);
            }
        }

        return(MStd);
    }

    // Convert from Eigen::Matrix to std::vector serializing by rows
    static std::vector<double> eigenToStd_MSd(const Eigen::MatrixXd &eigM)
    {
        unsigned int Nrows(eigM.rows());
        unsigned int Ncols(eigM.cols());

        std::vector<double> eigStd(Ncols*Nrows,0.);


        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigStd[row+Ncols*col]=eigM(row,col);
            }
        }

        return(eigStd);
    }
    static std::vector<float> eigenToStd_MSf(const Eigen::MatrixXf &eigM)
    {
        unsigned int Nrows(eigM.rows());
        unsigned int Ncols(eigM.cols());

        std::vector<float> eigStd(Ncols*Nrows,0.);


        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigStd[row+Ncols*col]=eigM(row,col);
            }
        }

        return(eigStd);
    }
    static std::vector<int> eigenToStd_MSi(const Eigen::MatrixXi &eigM)
    {
        unsigned int Nrows(eigM.rows());
        unsigned int Ncols(eigM.cols());

        std::vector<int> eigStd(Ncols*Nrows,0.);


        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigStd[row+Ncols*col]=eigM(row,col);
            }
        }

        return(eigStd);
    }

    // Convert from std::vector to Eigen::vector
    static Eigen::VectorXd stdToEigen_d(const std::vector<double> &eigStd)
    {
        return(Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(eigStd.data(), eigStd.size()));
    }
    static Eigen::VectorXf stdToEigen_f(const std::vector<float> &eigStd)
    {
        return(Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned>(eigStd.data(), eigStd.size()));
    }
    static Eigen::VectorXi stdToEigen_i(const std::vector<int> &eigStd)
    {
        return(Eigen::Map<const Eigen::VectorXi, Eigen::Unaligned>(eigStd.data(), eigStd.size()));
    }

    // Convert from std::vector<std::vector> to Eigen::Matrix, naive implementation
    static Eigen::MatrixXd stdToEigen_Md(const std::vector<std::vector<double>> &Mstd)
    {
        unsigned int Nrows(Mstd.size());
        unsigned int Ncols(Mstd[0].size());

        Eigen::MatrixXd eigM(Nrows,Ncols);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigM(row,col)=Mstd[row][col];
            }
        }

        return(eigM);
    }
    static Eigen::MatrixXf stdToEigen_Mf(const std::vector<std::vector<float>> &Mstd)
    {
        unsigned int Nrows(Mstd.size());
        unsigned int Ncols(Mstd[0].size());

        Eigen::MatrixXf eigM(Nrows,Ncols);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigM(row,col)=Mstd[row][col];
            }
        }

        return(eigM);
    }
    static Eigen::MatrixXi stdToEigen_Mi(const std::vector<std::vector<int>> &Mstd)
    {
        unsigned int Nrows(Mstd.size());
        unsigned int Ncols(Mstd[0].size());

        Eigen::MatrixXi eigM(Nrows,Ncols);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigM(row,col)=Mstd[row][col];
            }
        }

        return(eigM);
    }

    // Convert from std::vector serialized by rows to Eigen::Matrix
    static Eigen::MatrixXd stdToEigen_MSd(const std::vector<double> &eigStd,const unsigned int &Nrows, const unsigned int &Ncols)
    {

        Eigen::MatrixXd eigM(Ncols,Nrows);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigM(row,col)=eigStd[row+Ncols*col];
            }
        }

        return(eigM);
    }
    static Eigen::MatrixXf stdToEigen_MSf(const std::vector<float> &eigStd,const unsigned int &Nrows, const unsigned int &Ncols)
    {

        Eigen::MatrixXf eigM(Ncols,Nrows);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigM(row,col)=eigStd[row+Ncols*col];
            }
        }

        return(eigM);
    }
    static Eigen::MatrixXi stdToEigen_MSi(const std::vector<int> &eigStd,const unsigned int &Nrows, const unsigned int &Ncols)
    {

        Eigen::MatrixXi eigM(Ncols,Nrows);

        for(unsigned int row(0); row<Nrows; ++row)
        {
            for(unsigned int col(0); col<Ncols; ++col)
            {
                eigM(row,col)=eigStd[row+Ncols*col];
            }
        }

        return(eigM);
    }


    template<class T>
    static T SmoothInverse(const T &val, const T &fac_eps=100.,const char &method='l')
    {
        // Prevent division by 0
        T inv(1.),
          eps(fac_eps*std::numeric_limits<T>::epsilon());


        if(std::fabs(val)<eps)
        {
            if(method == 'l')
            {
                // Near zero, just change to linear linking the 2 branches of the inverse function
                inv = val;
            }
            else if(method == 'c')
            {
                // Near zero, link the 2 branches of the inverse with a cubic preserving continuity of derivative and curvature
                std::vector<T> Coeffs(2,0.),forcing(2,0.),matrix(4,0.);

                T C(-6./eps/eps/eps/eps);

                forcing[0] = (1./eps/eps-C)/eps;
                forcing[1] = 2./eps/eps/eps;

                matrix[0] = eps;
                matrix[1] = 1.;

                matrix[2] = 3.*eps;
                matrix[3] = 2.;

                matinv2x2(matrix);

                Coeffs[0] = matrix[0]*forcing[0]+matrix[1]*forcing[1];
                Coeffs[1] = matrix[2]*forcing[0]+matrix[3]*forcing[1];

                inv = Coeffs[0]*val*val*val+Coeffs[1]*val*val+C*val;
            }
            else if(method == 'd')
            {
                // Near zero, just clip the value to prevent overflow, but allow value discontinuity
                if(val >= 0.)
                    inv = 1./eps;
                else
                    inv = -1/eps;
            }
        }
        else
        {
            inv = 1./val;
        }

        return inv;
    }


    template<class T>
    static T SmoothAbsVal(const T &val, const T &fac_eps=100.)
    {
        // Build a continous in derivative and curvature approximation of absolute value function
        T absval(std::abs(val)),
          eps(fac_eps*std::numeric_limits<T>::epsilon());

        if(absval<eps)
        {
            std::vector<T> Coeffs(4,0.),forcing(4,0.),matrix(16,0.);

            forcing[0] = eps;
            forcing[1] = 1.;

            matrix[0 ] = eps*eps*eps*eps;
            matrix[1 ] = eps*eps*eps;
            matrix[2 ] = eps*eps;
            matrix[3 ] = 1.;

            matrix[4 ] = 4.*eps*eps*eps;
            matrix[5 ] = 3.*eps*eps;
            matrix[6 ] = 3.*eps;

            matrix[8 ] = 6.*eps*eps;
            matrix[9 ] = 3.*eps;
            matrix[10] = 1.;

            matrix[12] = 4.*eps;
            matrix[13] = 1.;

            matinv4x4(matrix);

            Coeffs[0] = matrix[0 ]*forcing[0]+matrix[1 ]*forcing[1]+matrix[2 ]*forcing[2]+matrix[3 ]*forcing[3];
            Coeffs[1] = matrix[4 ]*forcing[0]+matrix[5 ]*forcing[1]+matrix[6 ]*forcing[2]+matrix[7 ]*forcing[3];
            Coeffs[2] = matrix[8 ]*forcing[0]+matrix[9 ]*forcing[1]+matrix[10]*forcing[2]+matrix[11]*forcing[3];
            Coeffs[3] = matrix[12]*forcing[0]+matrix[13]*forcing[1]+matrix[14]*forcing[2]+matrix[15]*forcing[3];

            absval = Coeffs[0]*val*val*val*val+Coeffs[1]*val*val*val+Coeffs[2]*val*val+Coeffs[3];
        }

        return absval;
    }

    // Bisection method for a function with arguments
    template<class T>
    static T bisection(std::function<T(const T &,void *)> func,
                       T a,
                       T b,
                       std::pair<int, std::string> &error,
                       void *args4fun=nullptr,
                       const T &tol=1.e-4,
                       const unsigned int &maxIts=1000)
    {
        T fa = func(a,args4fun),
          fb = func(b,args4fun),
          fc = 0.;

        error.first = 0;
        error.second = "";

        if (fa*fb >= 0.)
        {
            error.second = "Problems in utilities::bisection(). No solution in the initial interval\n";
            error.second += "f(a) = " +NumToString(fa)+"\n";
            error.second += "f(b) = " +NumToString(fb)+"\n";
            error.first = -1;
            return 0.;
        }

        T c(a);
        unsigned int nits(0);
        while ((b-a) >= tol && nits < maxIts)
        {
            // Find middle point
            c = (a+b)/2;
            fc = func(b,args4fun);
            // Check if middle point is root
            if (abs(fc)<tol)
            {
                break;
            }
            // Decide the side to repeat the steps
            else if (fc*func(a,args4fun) < 0.)
            {
                b = c;
            }
            else
            {
                a = c;
            }
            ++nits;
        }
        return c;
    }

    // Bisection method for a function without arguments
    template<class T>
    static T bisection(std::function<T(const T &)> func,
                       T a,
                       T b,
                       std::pair<int, std::string> &error,
                       const T &tol=1.e-4,
                       const unsigned int &maxIts=1000)
    {
        T fa = func(a),
          fb = func(b);

        error.first = 0;
        error.second = "";

        if (fa*fb >= 0.)
        {
            error.second =  "Problems in utilities::bisection(). No solution in the initial interval\n";
            error.first = -1;
            return 0.;
        }
        T c(a);
        unsigned int nits(0);
        while ((b-a) >= tol && nits < maxIts)
        {
            // Find middle point
            c = (a+b)/2;

            // Check if middle point is root
            if (std::abs(func(c))<tol)
            {
                break;
            }
            // Decide the side to repeat the steps
            else if (func(c)*func(a) < 0.)
            {
                b = c;
            }
            else
            {
                a = c;
            }
            ++nits;
        }
        return c;
    }

    template<class T>
    static void computeErrorMeasures(const std::vector<T> &x,
                                     const std::vector<T> &uExact,
                                     const std::vector<T> &uNum,
                                     std::vector<T> &errorRMS,
                                     std::vector<T> &errorMAE,
                                     std::vector<T> &errorOverRMS,
                                     std::vector<T> &errorOverMAE,
                                     double &MAE,
                                     double &RMS)

    {
        size_t nn(x.size());

        if(nn!=uExact.size())
            MYTHROW("Problems in utilities::computeErrorMeasures(): Exact solution inconsistent with base!");

        if(nn!=uNum.size())
            MYTHROW("Problems in utilities::computeErrorMeasures(): Numerical solution inconsistent with base!");

        errorRMS.resize(nn,0.);
        errorOverRMS.resize(nn,0.);
        errorMAE.resize(nn,0.);
        errorOverMAE.resize(nn,0.);

        double RMS_SQ(0.);

        // Initialize the accumulator
        MAE = 0.;

        // Compute errors
        for(size_t inode(0); inode < nn; ++inode)
        {
            double du   = uExact[inode]-uNum[inode],
                   du2  = du*du,
                   fdu  = std::sqrt(du2);

            RMS_SQ += du2;

            MAE += fdu;

            errorRMS[inode] = du2;

            errorMAE[inode] = fdu;

        }

        RMS_SQ /= static_cast<double>(nn-1);
        MAE    /= static_cast<double>(nn);

        RMS = std::sqrt(RMS_SQ);

        for(size_t inode(0); inode < nn; ++inode)
        {
            errorOverRMS[inode] = errorRMS[inode]/RMS;
            errorOverMAE[inode] = errorMAE[inode]/MAE;
        }

    }

    template<class T>
    static void writeErrorMeasures(const std::vector<T> &x,
                                   const std::vector<T> &uExact,
                                   const std::vector<T> &uNum,
                                   const std::string &outFileName)
    {
        std::vector<double> errorRMS(0),
                            errorMAE(0),
                            errorOverRMS(0),
                            errorOverMAE(0);
        T RMS(0.), MAE(0.);

        computeErrorMeasures<T>(x,
                                uExact,
                                uNum,
                                errorRMS,
                                errorMAE,
                                errorOverRMS,
                                errorOverMAE,
                                MAE,
                                RMS);
        // Write to file
        std::ofstream outFile(outFileName.c_str());
        outFile << std::scientific << std::setprecision(8);
        outFile << "RMS\n"<<RMS<<"\nMAE\n"<<MAE<<"\n";
        outFile << "x\t abs Error RMS \t (abs Error RMS)/RMS \t";
        outFile << " abs Error MAE \t (abs Error MAE)/MAE \n";
        for(size_t inode(0); inode < x.size(); ++inode)
        {
            outFile << x[inode] << "\t"
                    << errorRMS[inode] << "\t"
                    << errorOverRMS[inode] << "\t"
                    << errorMAE[inode] << "\t"
                    << errorOverMAE[inode] << "\n";
        }

        outFile.close();
    }

    template<class T>
    static void writeErrorMeasures(const std::string &dataExact,
                                   const std::string &dataNum,
                                   const std::pair<size_t, size_t> &colsToReadExact=std::make_pair(0,1),
                                   const size_t &thingsToReadExact=2,
                                   const std::pair<size_t, size_t> &colsToReadNum=std::make_pair(0,1),
                                   const size_t &thingsToReadNum=2)
    {

        std::vector<double> xExact(0),
                            xNum(0),
                            uExact(0),
                            uNum(0),
                            errorRMS(0),
                            errorMAE(0),
                            errorOverRMS(0),
                            errorOverMAE(0);

        readFile(dataNum,xNum,uNum,colsToReadNum,thingsToReadNum);
        readFile(dataExact,xExact,uExact,colsToReadExact,thingsToReadExact);


        if(xNum != xExact)
            MYTHROW("Problems in utilities::writeErrorMeasures(): Bases are not consistent!");


        T RMS(0.), MAE(0.);

        computeErrorMeasures<T>(xExact,
                                uExact,
                                uNum,
                                errorRMS,
                                errorMAE,
                                errorOverRMS,
                                errorOverMAE,
                                MAE,
                                RMS);
        // Write to file
        std::string output = "Errors_" + utilities::RemoveExtension(dataNum) +
                "_" + utilities::RemoveExtension(dataExact)  + ".dat";

        std::ofstream outFile(output.c_str());
        outFile << std::scientific << std::setprecision(8);
        outFile << "RMS\n"<<RMS<<"\nMAE\n"<<MAE<<"\n";
        outFile << "x\t abs Error RMS \t (abs Error RMS)/RMS \t";
        outFile << " abs Error MAE \t (abs Error MAE)/MAE \n";
        for(size_t inode(0); inode < xExact.size(); ++inode)
        {
            outFile << xExact[inode] << "\t"
                    << errorRMS[inode] << "\t"
                    << errorOverRMS[inode] << "\t"
                    << errorMAE[inode] << "\t"
                    << errorOverMAE[inode] << "\n";
        }

        outFile.close();
    }

    template<class T>
    static bool isCloseToZero(const T &val,const T &eps=1.e3*std::numeric_limits<T>::epsilon())
    {
        return std::fabs(val)<eps;
    }


    // Convolution in the time domain
    template<class T>
    static std::vector<T>  convolution(const std::vector<T> &f,const std::vector<T>  &g)
    {
        const size_t nf = f.size(),
                     ng = g.size(),
                     n  = nf + ng - 1;

        std::vector<T> out(n,0);
        for(size_t i(0); i < n; ++i)
        {
            const size_t jmn = (i >= ng - 1)? i - (ng - 1) : 0;
            const size_t jmx = (i <  nf - 1)? i            : nf - 1;
            for(size_t j(jmn); j <= jmx; ++j)
            {
                out[i] += (f[j] * g[i - j]);
            }
        }
        return out;
    }


    // Laplace transform from https://www.codeproject.com/Articles/25189/Numerical-Laplace-Transforms-and-Inverse-Transform#TheInverseLaplaceTransform3
    template<class T>
    static T Factorial(const size_t &N)
    {
        T x = 1;
        if (N > 1)
        {
            for (size_t i = 2; i <= N; ++i)
                x *= i;
        }
        return x;
    }


    template<class T>
    static std::vector<T> InitStehfest(const size_t &N)
    {

        // Make sure that thigs are powers of two
        size_t N2 = N / 2;
        size_t NV = 2 * N2;

        std::vector<T>  V(NV,0.);// Stehfest coefficients
        short int sign = 1;
        if (!isEven(N2))
            sign = -1;
        for (size_t i = 0; i < NV; ++i)
        {
            size_t kmin = (i + 2) / 2;
            size_t kmax = i + 1;

            if (kmax > N2) kmax = N2;

            V[i] = 0.;
            sign = -sign;
            for (size_t k = kmin; k <= kmax; ++k)
            {
                V[i] += (std::pow(k, N2) / Factorial<T>(k)) * (Factorial<T>(2 * k)
                     / Factorial<T>(2 * k - i - 1)) / Factorial<T>(N2 - k)
                     / Factorial<T>(k - 1) / Factorial<T>(i + 1 - k);
            }
            V[i] = sign * V[i];
        }
        return V;
    }

    template<class T>
    static T InverseLaplaceTransform(std::function<T(const T &)> f,const  T &t,const size_t &N=14)
    {
        T ln2 = std::log(2.0);
        T ln2t = ln2 / t;
        T x = 0;
        T y = 0;
        std::vector<T>  V = InitStehfest<T>(N);
        for (size_t i = 0; i < V.size(); i++)
        {
            x += ln2t;
            y += V[i] * f(x);
        }
        return ln2t * y;
    }



    template<class T>
    static T LaplaceTransform(std::function<T(const T &)> F,const T &s,const size_t &DefaultIntegralN = 5000)
    {
        T du = 0.5 / (T)DefaultIntegralN;
        T y =  - F(0) / 2.0;
        T u = 0.;
        T limit = 1.0 - 1.e-10;
        while (u < limit)
        {
            u += du;
            y += 2.0 * std::pow(u, s - 1.) * F(-std::log(u));
            u += du;
            y += std::pow(u, s - 1.) * F(-std::log(u));
        }
        return 2.0 * y * du / 3.0;
    }

    template<class T>
    static bool isPowerOfTwo(T x)
    {
        return (x != 0) && ((x & (x - 1)) == 0);
    }

    // sign function from https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c      
    template <typename T> inline constexpr
    static short int signum(const T &x, std::false_type is_signed) {
        return T(0) < x;
    }

    template <typename T> inline constexpr
    static short int signum(const T &x, std::true_type is_signed) {
        return (T(0) < x) - (x < T(0));
    }

    template <typename T> inline constexpr
    static short int signum(const T &x) {
        return signum(x, std::is_signed<T>());
    }

    template<class T>
    static T logarithmicMean(const T &a,const T &b)
    {
       if(!(a>0.) || !(b>0.)) MYTHROW("Negative or zero values in utilities::logarithmicMean()");

        T logmean = 0.;
        T diff = a-b;

        if(std::abs(diff)<1.e-2)
        {
            logmean = diff/(log(a)-log(b));
        }
        else
        {
            T mean  = 0.5*(a+b);

            T fac = 1.;
            fac += diff*diff/(12.*mean*mean);
            fac += diff*diff*diff*diff/(80.*mean*mean*mean*mean);

            logmean = mean/fac;
        }

        return logmean;
    }

    // Naive and innefficient binomial coefficient calculation
    template<class T>
    static T BinomialCoeffs(const size_t &num,const size_t &den)
    {
        T res(1.);
        size_t numerator   = num,
               denominator = den;

        //default values
        if(denominator==numerator || denominator==0 || numerator==0)
            return res;


        // Since C(n, k) = C(n, n-k)
        if ( denominator > (numerator - denominator) )
            denominator = numerator - denominator;


        // Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
        while (denominator>=1)
        {

            res *= numerator;
            res = res / denominator;

            denominator--;
            numerator--;
        }

        return res;
    }

    // Bernstein matrix filling function
    static Eigen::MatrixXd BernsteinMatrix (const size_t &order)
    {
        size_t size = order+1;
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(size,size);

        for(size_t idx=0; idx<size;++idx)
        {
            size_t counter = 0;
            for(size_t jdx=counter; idx<size;++idx)
            {
                size_t omc = order-counter,
                       jmc = jdx-counter;
                double B1 = BinomialCoeffs<double>(order,idx),
                       B2 = BinomialCoeffs<double>(omc,jmc);

                B(idx,jdx) = std::pow(-1.,jdx-counter)*B1*B2;
            }

            ++counter;
        }

        return B;

    }

    template<class T>
    static Eigen::VectorXd BernsteinBasis(const T &t,const size_t &order)
    {
        size_t size = order+1;
        Eigen::VectorXd V = Eigen::MatrixXd::Zero(size,size);

        for(size_t idx=0; idx<size;++idx)
        {
            V(idx) = std::pow(t,idx);
        }

        return BernsteinMatrix(order)*V;

    }


    // Evaluate a single knot Bezier curve as a function of the interval t E [0,1]
    static std::vector<std::vector<double>> BezierCurve(const std::vector<std::vector<double>> &controlPoints, const size_t &Tsamples=100)
    {
        Eigen::MatrixXd  controlPointsEig   = stdToEigen_Md(controlPoints);

        std::vector<std::vector<double>> curvePoints(0);

        size_t order = controlPoints.size();

        for(size_t idx(0); idx < Tsamples; ++idx)
        {
            double t = static_cast<double>(idx)/static_cast<double>(Tsamples-1);

            Eigen::VectorXd pointsCurve = controlPointsEig.transpose()*BernsteinBasis(t,order);

            curvePoints.push_back(eigenToStd_d(pointsCurve));
        }

        return curvePoints;

    }
    template<class T>
    static std::vector<T> movingStatistic(std::function<T(const std::vector<T> &)> func,const std::vector<T> &signal,const size_t &nSamples)
    {
        std::vector<T> res(0);

        bool endLoop(false);

        size_t N = signal.size();

        for(size_t i=0; i < N;++i)
        {
            std::vector<T> samples(0);

            for(size_t s=0; s < nSamples;++s)
            {
                size_t localIdx = i+s;
                if(localIdx == N)
                {
                    endLoop=true;
                    break;
                }
                samples.push_back(signal[localIdx]);
            }

            if(endLoop)break;

            res.push_back(func(samples));
        }

        return res;
    }

    template<class T>
    static std::vector<T> movingAverage(const std::vector<T> &signal,const size_t &nSamples)
    {
        std::function<T(const std::vector<T> &)> func = std::bind(utilities::ComputeMean<T>,std::placeholders::_1);
        return movingStatistic(func,signal,nSamples);
    }

    template<class T>
    static std::vector<T> movingMedian(const std::vector<T> &signal,const size_t &nSamples)
    {
        std::function<T(const std::vector<T> &)> func = std::bind(utilities::ComputeMedian<T>,std::placeholders::_1);
        return movingStatistic(func,signal,nSamples);
    }

    // Check parity. Should work for ints and size_t's
    template<class T>
    static bool isEven(const T &number)
    {
        bool even = false;
        if(number & 1 == 0) even = true;
        return even;
    }



    // Parallel loops: Ref https://stackoverflow.com/questions/36246300/parallel-loops-in-c/36246386
    // Edited to use size_t instead of unsigned and passing by reference of indices
    // Fixed a bug related to specifying the start index as input
    static void threadParFor(const size_t &start, const size_t &end, std::function<void(const size_t &i)> fn, bool par = true)
    {

        // Internal loop
        auto int_fn = [&fn](const size_t &int_start, const size_t &seg_size)
        {
            for(size_t j = int_start; j < int_start+seg_size; j++)
            {
                fn(j);
            }
        };

        // Sequenced for
        size_t total_length = end - start;
        if(!par)
        {
            return int_fn(start, total_length);
        }

        // Get number of threads
        size_t nb_threads_hint = std::thread::hardware_concurrency(),
               nb_threads      = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

        // Calculate segments
        size_t seg      = total_length/nb_threads,
               last_seg = seg + total_length%nb_threads;

        // Launch threads - parallel for
        auto threads_vec = std::vector<std::thread>();
        threads_vec.reserve(nb_threads);
        for(size_t k = 0; k < nb_threads-1; ++k)
        {
            size_t current_start = start+seg*k;
            threads_vec.emplace_back(std::thread(int_fn, current_start, seg));
        }
        {
            size_t current_start = start+seg*(nb_threads-1);
            threads_vec.emplace_back(std::thread(int_fn, current_start, last_seg));
        }
        for (auto& th : threads_vec)
        {
            th.join();
        }
    }


    static void asyncParFor(const size_t &start,const size_t &end, std::function<void(const size_t &i)> fn, bool par = true)
    {

        // Internal loop
        auto int_fn = [&fn](const size_t &int_start,const size_t &seg_size)
        {
            for (size_t j = int_start; j < int_start+seg_size; j++)
            {
                fn(j);
            }
        };

        // Sequenced for
        size_t total_length = end - start;
        if(!par)
        {
            return int_fn(start, total_length);
        }

        // Get number of threads
        size_t nb_threads_hint = std::thread::hardware_concurrency();
        size_t nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

        // Calculate segments

        size_t seg = total_length/nb_threads;
        size_t last_seg = seg + total_length%nb_threads;

        // Launch threads - parallel for
        auto fut_vec = std::vector<std::future<void>>();
        fut_vec.reserve(nb_threads);
        for(size_t k = 0; k < nb_threads-1; ++k)
        {
            size_t current_start = start+seg*k;
            fut_vec.emplace_back(std::async(int_fn, current_start, seg));
        }
        {
            size_t current_start = start+seg*(nb_threads-1);
            fut_vec.emplace_back(std::async(std::launch::async, int_fn, current_start, last_seg));
        }
        for (auto& th : fut_vec)
        {
            th.get();
        }
    }

    // Repurposes the for_each for loop parallelization: Ref https://stackoverflow.com/questions/36246300/parallel-loops-in-c/36246386
    // UNABLE TO COMPILE IT, NEEDS TBB > V2018 AND I'M UNABLE TO INSTALL IT
//    static void forEach(const size_t &start, const size_t &end, std::function<void(const size_t &i)> fn, bool par = true)
//    {
//        std::vector<size_t> idxList(end-start) ;
//        std::iota (std::begin(idxList), std::end(idxList), start);

////        auto int_fn = [&fn](auto&& item)
////        {
////            fn(item);
////        };

//        if(par)
//        {
//            std::for_each(std::execution::par_seq,idxList.begin(),idxList.end(),fn);
//        }
//        else
//        {
//            std::for_each(std::execution::seq,idxList.begin(),idxList.end(),fn);
//        }
//    }

//    static void forEachAsync(const size_t &start, const size_t &end, std::function<void(const size_t &i)> fn, bool par = true)
//    {
//        std::vector<size_t> idxList(end-start) ;
//        std::iota (std::begin(idxList), std::end(idxList), start);

////        auto int_fn = [&fn](auto&& item)
////        {
////            fn(item);
////        };

//        if(par)
//        {
//            std::for_each(std::execution::par_unseq,idxList.begin(),idxList.end(),fn);
//        }
//        else
//        {
//            std::for_each(std::execution::unseq,idxList.begin(),idxList.end(),fn);
//        }
//    }


    // Perform richardson extrapolation, compute grid convergence metrics and order of error.
    // Based on https://www.grc.nasa.gov/www/wind/valid/tutorial/verify.f90
    // Follow the ref: Is your mesh refined enough? Estimating discretization error using GCI
    // Leonard E. Schwer, internal report, Schwer Engineering & Consulting Services, Windsor, CA, USA, 2008
    // for the case of non uniform refinement ratios
    static void  gridConvergence(const std::vector<size_t> &gridSizes,
                                 const std::vector<double> &outputVals,
                                 double &convergenceOrder,
                                 double &RichardsonExtrapolation,
                                 std::vector<double> &gcif,
                                 std::vector<double> &asymptoticRatio,
                                 const size_t &dimension=1)
    {

        // First sort in case the input is not properly ordered
        Sort2Vectors(gridSizes,outputVals);

        //Write Header

        std::cout <<" " << std::endl;
        std::cout <<"--- VERIFY: Performs verification calculations ---" << std::endl;

        // Compute the grid refinement ratio, r, between each pair.
        std::vector<double> ratios(0);
        for(size_t n = 0; n < gridSizes.size()-1; ++n)
        {
            double ratio    = double(gridSizes[n+1])/double(gridSizes[n]);
            double effRatio = std::pow(ratio,1./double(dimension));
            ratios.push_back(effRatio);
        }

        // Check if the ratios are uniform
        bool isUniform = false;
        if ( std::adjacent_find( ratios.begin(), ratios.end(), std::not_equal_to<>() ) == ratios.end() )
        {
            std::cout << "All refinement ratios are equal each other" << std::endl;
            isUniform = true;
        }


        double fRatio = ( outputVals[2] - outputVals[1] ) / ( outputVals[1] - outputVals[0] );
        if(isUniform)
        {
            // Estimate the order of convergence using the first three data pairs
            // and assuming that the grid refinement ratio is constant, r[0] = r[1].
            // This is done using Eqn. 5.10.6.1 of (Roache, 1998).
            convergenceOrder = std::log( fRatio )  /  std::log( ratios[0] );
        }
        else
        {
            // Estimate the order of convergence using the first three data pairs
            // This is done using Eqn. 10 of (Schwer, 2008).
            int    s = signum(fRatio);

            auto toSolve = [&fRatio,&s,&ratios](const double&convOrder)
            {
                double rsRatio = (std::pow(ratios[0],convOrder) - double(s) ) / ( std::pow(ratios[1],convOrder) - double(s) ),
                       q       = std::log(rsRatio),
                       logF    = std::log(fRatio),
                       p       = std::abs(logF+q)/std::log(ratios[0]);
                return p;
            };

            double qLow  = 0.,
                   qHigh = 10.;
            std::pair<int, std::string> error;
            convergenceOrder = bisection<double>(toSolve,qLow,qHigh,error);

            if(error.first == -1)  MYTHROW(error.second);

        }

        std::cout << " " << std::endl;
        std::cout << "Order of convergence using first three finest grids " << std::endl;
        std::cout << "and assuming constant grid refinement " << std::endl;
        std::cout << "Order of Convergence, p = " << convergenceOrder << std::endl;


        //Perform Richardson extrapolation on the finest meshes to estimate a zero grid value of f.

        double factor = std::pow(ratios.back(),double(convergenceOrder));
        RichardsonExtrapolation =  ( factor*outputVals[outputVals.size()-2] - outputVals.back() ) / ( factor - 1.0 );

        std::cout << " " << std::endl;
        std::cout << "Richardson Extrapolation: Use above order of convergence" << std::endl;
        std::cout << "and first and second finest grids (Eqn. 5.4.1) " << std::endl;
        std::cout << "Estimate to zero grid value, f_exact = " << RichardsonExtrapolation << std::endl;

        //Compute Grid Convergence Index (GCI) for each fine grid using Eqn. 5.6.1
        //from Roache's book. Use safety factor as recommended on page 123.
        double fsafe = 3.;
        if( gridSizes.size() >= 3 )
        {
            fsafe = 1.25;
        }

        gcif.clear();
        for(size_t n = 0; n < gridSizes.size()-1; ++n)
        {
            double e   = std::abs( ( 1. - outputVals[n]/outputVals[n+1]) ),
                   val = fsafe * e;

            val /= std::pow(ratios[n],convergenceOrder) - 1.0;

            gcif.push_back(val);
        }

        std::cout << " " << std::endl;
        std::cout << " Grid Convergence Index on fine grids. Uses p from above." << std::endl;
        std::cout << " Factor of Safety = " << fsafe << std::endl;
        std::cout << " " << std::endl;
        std::cout << " Grid refinement            " << std::endl;
        std::cout << " Step          Ratio, r          GCI(%)" << std::endl;
        for(size_t n = 0; n < gridSizes.size()-1; ++n)
        {
            std::cout << n <<" "<< n+1 << "          " << ratios[n]<< "          " << gcif[n]*100.0 << std::endl;
        }

        // Examine if asymptotic range has been achieved by checking if ratio
        // of Eqn. 5.10.5.2 of Roache's book is one.

        if( gridSizes.size() >= 3 )
        {

            std::cout << " "  << std::endl;
            std::cout << " Checking for asymptotic range using Eqn. 5.10.5.2." << std::endl;
            std::cout << " A ratio of 1.0 indicates asymptotic range." << std::endl;
            std::cout << " " << std::endl;
            std::cout << " Grid Range          Ratio" << std::endl;
            asymptoticRatio.clear();
            for(size_t n = 0; n < gridSizes.size()-2; ++n)
            {
                double aRatio = std::pow(ratios[n],convergenceOrder) * gcif[n] / gcif[n+1];
                asymptoticRatio.push_back(aRatio);
                std::cout << n << "-" << n+1 << " " << n+1 << "-"<< n+2 << "          " << aRatio << std::endl;
            }

        }



        //Write Trailer.

        std::cout << " " << std::endl;
        std::cout << "--- End of VERIFY ---" << std::endl;
        std::cout << " " << std::endl;
    }

    // Overloaded version that reads from an inputfile
    static void  gridConvergence(const std::string &inputFile,
                                 double &convergenceOrder,
                                 double &RichardsonExtrapolation,
                                 std::vector<double> &gcif,
                                 std::vector<double> &asymptoticRatio,
                                 const size_t &dimension=1)
    {
         std::vector<double> gridSizesD(0),outputVals(0);

         // Read from input file
         readFile(inputFile,gridSizesD,outputVals);

         // Call basic method
         gridConvergence(ContainerCaster<size_t,double>(gridSizesD),outputVals,convergenceOrder,RichardsonExtrapolation,gcif,asymptoticRatio,dimension);
    }


};





#endif
