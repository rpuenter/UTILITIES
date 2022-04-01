#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <iomanip>

#include "utilities.h"

void debugUtil::wait()
{
    while(isDebug)
    {

    }
}

void utilities::readFile(const std::string &File_name,
                         std::vector<double> &x,
                         std::vector<double> &u,
                         const std::pair<size_t,size_t> &colsToRead,
                         const size_t &thingsToRead)
{

    if(std::max(colsToRead.first,colsToRead.second)>thingsToRead)
        MYTHROW("Trying to read more things than there are in utilities::readFile()");

    std::ifstream File(File_name.c_str());

    x.clear();
    u.clear();

    if(File.good())
    {
        std::string Line;


        int error(0); // Error flag

        std::vector<std::string> readThings(0);
        while(!File.eof())
        {
            std::getline(File,Line);
            if(!Line.empty() && !utilities::isComment(Line))
            {
                utilities::SplitStringN(Line,thingsToRead,readThings,error);
                x.push_back(utilities::strToDouble(readThings[colsToRead.first]));
                u.push_back(utilities::strToDouble(readThings[colsToRead.second]));
            }
        }

    }
    else
    {
        std::string mssg("Error in utilities::readFile(): No input file in this directory!\n");
        throw(mssg);
    }


    File.close();

}


/*------------------------------------------------------------------------------
 Removes white spaces, ",", ";" and tabs from the begining of the std::string and
 at the end.
-----------------------------------------------------------------------------*/
void utilities::CleanString(std::string &input)
{
    if(input.empty()) return;
    while (input[0]==' ' || input[0]==9 || input[0]=='\n' || input[0]==',' ||
           input[0]==';')
    {
        input = input.substr(1);
        if(input.empty()) break;
    }

    if(input.empty()) return;
    while (input[input.size()-1]==' ' ||
           input[input.size()-1]==9   ||
           input[input.size()-1]==',' ||
           input[input.size()-1]==';' ||
           input[input.size()-1]=='\n')
    {
        input = input.substr(0,input.size()-1);
        if(input.empty()) break;
    }
}


bool utilities::CheckExtension(const std::string &input, const std::string &ext)
{

    /*
  Returns true if "ext" corresponds to input's file extension
*/

    const std::string point(".");
    std::string buffer(input);
    std::string::size_type Idx = buffer.find_last_of(point);

    if(Idx == std::string::npos)
    {
        MYTHROW("No file extension found!");
    }
    else
    {

        buffer = buffer.substr(Idx+1);

        if(buffer==ext)
            return(true);
        else
            return(false);
    }

}

std::string utilities::RemoveExtension(const std::string &input)
{

    /*
  Removes the extension in the file name "input"
*/

    const std::string point(".");
    std::string buffer(input);
    std::string::size_type Idx = buffer.find_last_of(point);

    if(Idx == std::string::npos)
    {
       return(input);
    }
    else
    {

        buffer = buffer.substr(0,Idx);

        return(buffer);

    }

}


void utilities::FileToStream(std::string File_name, std::stringstream &outstream)
{

    std::ifstream File(File_name.c_str());
    std::string Line;

    if(!File) MYTHROW("Problems opening " + File_name + "!\n");

    while(!File.eof())
    {

        std::getline(File,Line);
        outstream << Line << std::endl;

    }

}

bool utilities::isComment(const std::string &line)
{
    bool isC(false);
    const std::string commentChars("#/!");
    for(size_t idx(0); idx < commentChars.size();++idx)
    {
        if(line[0] == commentChars[idx])
        {
            isC=true;
            break;
        }
    }

    return(isC);

}

void utilities::SplitStringN(std::string Input,  size_t N,
                             std::vector<std::string> & Out, int & error)
{
    /*

  Split Input string into N using blanks, ",", ";" and tabs as
  delimiters. Checks if delimiters exist

  Needs exceptions implemented

*/
    const std::string delimiters(" ,;	");
    SplitStringN(Input,N,Out,delimiters,error);
}

void utilities::SplitStringN(std::string Input, size_t N,
                             std::vector<std::string> & Out, const std::string &delimeters, int & error)
{

    /*

  Split Input std::string into N using blanks the input delimiters. Checks if delimiters exist

  Needs exceptions implemented

*/


    std::string buffer(Input);
    std::string::size_type Idx;
    CleanString(buffer);

    size_t counter(0);

    error = 0;

    Idx = buffer.find_first_of(delimeters);
    if(Idx == std::string::npos)
    {

        Out.resize(1);
        std::cout << "No delimiters found!\n";
        error = -1;
        Out[0]= buffer;

    }
    else
    {
        Out.resize(N);
        for(counter = 0; counter < (N-1); ++counter)
        {
            Idx = buffer.find_first_of(delimeters);

            Out[counter]= buffer.substr(0,Idx);
            buffer = buffer.substr(Idx);
            CleanString(buffer);

        }

        Out[counter]= buffer;
    }


}


bool utilities::replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}




