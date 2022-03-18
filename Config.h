#pragma once
/****************************************************************************
*   日期:  2014-12-5
*   目的:  读取配置文件的信息，以map的形式存入
*   要求:  配置文件的格式，以#作为行注释，配置的形式是key = value，中间可有空
格，也可没有空格
*****************************************************************************/
#define COMMENT_CHAR '#'
#include <string>
#include <map>

//using namespace std;

bool ReadConfig(const std::string & filename, std::map<std::string, std::string> & m);

void PrintConfig(const std::map<std::string, std::string> & m);
