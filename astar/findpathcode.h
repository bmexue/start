#pragma once
#include <vector>
#include <deque>
#include<queue>
#include <iostream>
#include <queue>
#include <deque>
#include <vector>
#include <functional>
#include <cmath>

using namespace std ;



typedef std::vector<POINT> PRIDEQ;
 //typedef priority_queue<int,vector<int>,greater<int> > PRIDEQ;


struct tagNode
{
	POINT pFather;    //  父亲节点
	bool  bAnnalyze;  //  是否被遍历过
	int   nGot;
	int   nYugu;

	int   nStat ;  // -1 未知，0，已经分析过，   1Open状态


	tagNode()
	{
		bAnnalyze = false;
		pFather.x = -1;
		pFather.y =-1;
		nGot = -1;
		nYugu = -1;
		nStat = -1;
	}
};


int GetIndex(int i, int j, int nNum );

bool CheckPoint(  int nIndex, int * pnnMap,tagNode *pOutList );

// 走法难度都是  10  20  30  40      斜走  14  28  42  56
// 走一半都是   5 10 15 20       7  14  21  28


int GetHWPart( int nIndex, int * pnnMap );

int GetXPart( int nIndex,int * pnnMap );


void SetOneNewPoint(int nNewIndex,int nCurIndex,tagNode *pOutList,bool bHw,int * pnnMap ,PRIDEQ & openNodes );


int AnnalyePoint( int i, int j,int nDx, int nDy,int * pnnMap,tagNode *pOutList, POINT pEnd, PRIDEQ & openNodes);



int  FindPathByW( int * pnnMap, int nDx, int nDy, POINT pStart, POINT pEnd,tagNode *pOutList,bool bBfs = true,double lfRate=1.0);
