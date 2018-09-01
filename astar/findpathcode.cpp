#include "stdafx.h"
#include "findpathcode.h"




int GetIndex(int i, int j, int nNum )
{
	return i*nNum + j;
}

bool CheckPoint(  int nIndex, int * pnnMap,tagNode *pOutList )
{
	 if( pnnMap[nIndex] != 4 )
	 {
		 if( pOutList[nIndex].nStat != 0 )
		 {
			 return true;
		 }
	 }
	 return false;
}

// 走法难度都是  10  20  30  40      斜走  14  28  42  56
// 走一半都是   5 10 15 20       7  14  21  28


int GetHWPart( int nIndex, int * pnnMap )
{
	return 5  + 5 * pnnMap[nIndex];
}

int GetXPart( int nIndex,int * pnnMap )
{
	return 7  + 7 * pnnMap[nIndex];
}

POINT g_pEnd;
int   g_nDx;
double g_lfrate;
void SetOneNewPoint(int nNewIndex,int nCurIndex,tagNode *pOutList,bool bHw,int * pnnMap ,PRIDEQ & openNodes )
{
	if( pOutList[nNewIndex].nStat == 0 )
	{
		return ;
	}
	int i = nNewIndex/g_nDx;
	int j = nNewIndex%g_nDx;

	int iOld = nCurIndex/g_nDx;
	int jOld = nCurIndex%g_nDx;

	int nDis = -1;
	if( bHw )
	{
		nDis = GetHWPart(nCurIndex,pnnMap ) +  GetHWPart(nNewIndex ,pnnMap) ;
	}
	else
	{
		nDis = GetXPart(nCurIndex,pnnMap ) +  GetXPart(nNewIndex ,pnnMap) ;
	}

	if( pOutList[nNewIndex].nGot == -1 )
	{
		pOutList[nNewIndex].nGot = pOutList[nCurIndex].nGot + nDis;

		pOutList[nNewIndex].pFather.x = iOld;
		pOutList[nNewIndex].pFather.y = jOld;
	}
	else
	{
		int nNewPath = pOutList[nCurIndex].nGot  + nDis ;
		if( pOutList[nNewIndex].nGot > nNewPath )
		{
			pOutList[nNewIndex].nGot = nNewPath;

			pOutList[nNewIndex].pFather.x = iOld;
			pOutList[nNewIndex].pFather.y = jOld;
		}
	}
	if( pOutList[nNewIndex].nStat == -1 ) // 改点是新点，存放到开放列表里去
	{
		POINT pt;
		pt.x = i;  pt.y=j;
		openNodes.push_back(pt);
	}
	pOutList[nNewIndex].nStat = 1; // 该节点被打开了

	// 这里是精髓，如何预估方法
	{
        int dx = abs(i - g_pEnd.x) ;
		int dy = abs(j - g_pEnd.y) ;
		int nMax = -1;
		int nMin = -1;
		if( dx >=dy )
		{
			nMax = dx;
			nMin = dy;
		}
		else
		{
			nMax = dy;
			nMin = dx;
		}

		double lfPath = 10* (nMax-nMin) +   14*nMin ;

		pOutList[nNewIndex].nYugu =   lfPath * g_lfrate ;
	}
}

int AnnalyePoint( int i, int j,int nDx, int nDy,int * pnnMap,tagNode *pOutList, POINT pEnd, PRIDEQ & openNodes)
{
	if ( i == pEnd.x && j == pEnd.y )
	{
		return 0;  //  计算结束
	}

    int ii = -1;
	int jj = -1;

	int nCurIndex = GetIndex(i,j,nDx);

	// 分析点

	// 左上
	ii=i-1;
	jj=j-1;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,false,pnnMap,openNodes);	
	}

	// 上
	ii=i;
	jj=j-1;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,true,pnnMap,openNodes);
	}

	// 右上
	ii=i+1;
	jj=j-1;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,false,pnnMap,openNodes);
	}

	// 左
	ii=i-1;
	jj=j;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,true,pnnMap,openNodes);
	}

	// 右
	ii=i+1;
	jj=j;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,true,pnnMap,openNodes);;
	}

	// 左下
	ii=i-1;
	jj=j+1;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,false,pnnMap,openNodes);
	}

	// 正下
	ii=i;
	jj=j+1;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,true,pnnMap,openNodes);
	}

	// 右下
	ii=i+1;
	jj=j+1;
	if( ii >=0 && ii<nDx && jj>=0 && jj<nDy && CheckPoint( GetIndex(ii,jj,nDx),pnnMap,pOutList ) )
	{
		int nNewIndex = GetIndex(ii,jj,nDx);
		SetOneNewPoint( nNewIndex,nCurIndex,pOutList,false,pnnMap,openNodes);
	}
	pOutList[nCurIndex].bAnnalyze = true; // 不是算法的， 只是为了统计方便
	pOutList[nCurIndex].nStat     = 0;  // 该点是关闭节点了

	return 1;
}


int FindMinNode(PRIDEQ & nOpenNodes, int nDx, bool bBfs,tagNode *pOutList)
{
	// 在开放节点中寻找最优的点
	int nMinPath = 0x0fffffff;
	int nDex = -1;
	PRIDEQ::iterator itr = nOpenNodes.begin();
	for(  ; itr != nOpenNodes.end(); itr++)
	{
		int nCurIndexT =  GetIndex(itr->x, itr->y,nDx);
		if( nDex == -1 )
		{
			nDex =nCurIndexT;
			if( !bBfs)
			{
				nMinPath = pOutList[nDex].nGot + pOutList[nDex].nYugu;
			}
		}
		else
		{
			if( bBfs )
			{
				if( pOutList[nDex].nGot > pOutList[nCurIndexT].nGot )   // 这里只比较了已经消耗的路径，并且寻找最小的
				{
					nDex =nCurIndexT;
				}	
			}
			else  // A *
			{
				if( pOutList[nDex].nYugu != -1 && pOutList[nCurIndexT].nYugu != -1 )
				{
					//int nTmp1 = pOutList[nDex].nGot + pOutList[nDex].nYugu;
					int nTmpPath = pOutList[nCurIndexT].nGot + pOutList[nCurIndexT].nYugu;
					if( nTmpPath < nMinPath )   // 这里只比较了已经消耗的路径，并且寻找最小的
					{
						nDex = nCurIndexT;
						nMinPath = nTmpPath;
					}
				}
			}

		}
	}
	return nDex;
}

int  FindPathByW( int * pnnMap, int nDx, int nDy, POINT pStart, POINT pEnd,tagNode *pOutList, bool bBfs ,double lfRate)
{
	g_lfrate = lfRate;
	g_pEnd = pEnd;
	g_nDx = nDx;

	int nCurIndex = GetIndex(pStart.x,pStart.y,nDx);
	pOutList[nCurIndex].bAnnalyze = true;
	pOutList[nCurIndex].nGot = 0;
	pOutList[nCurIndex].nStat = 0;
	pOutList[nCurIndex].nYugu = -1;
	pOutList[nCurIndex].pFather.x = -1;
	pOutList[nCurIndex].pFather.y = -1;


	PRIDEQ  nOpenNodes; //最小堆

	int nx = pStart.x;
	int ny = pStart.y;

	int nres = -1;
	while (true )
	{
		// 得到了若干个Open节点，所谓Open节点，就是
	   nres  = AnnalyePoint( nx,ny,nDx,nDy, pnnMap,pOutList,pEnd, nOpenNodes);
	   if( nres == 0 )
	   {
		   break;
	   }	
	  
	    // 删除当前节点  因为这个节点进入了Close状态
	   PRIDEQ::iterator itr = nOpenNodes.begin();
	   for(  ; itr != nOpenNodes.end(); itr++)
	   {
		   if( itr->x == nx && itr->y == ny)
		   {
			   nOpenNodes.erase(itr);	
			   break;
		   }
	   }

	   if( nOpenNodes.size() ==0 )
	   {
		   break;
	   }
	   // 在开放节点中寻找最优的点
	   int nDex =  FindMinNode(nOpenNodes, nDx, bBfs,pOutList);	 
	   nx = nDex/nDx;
	   ny = nDex%nDx;
	   int t= -1;
	}
	return nres;
}