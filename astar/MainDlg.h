// MainDlg.h : interface of the CMainDlg class
//
/////////////////////////////////////////////////////////////////////////////

#pragma once

#include "resource.h"
#include "aboutdlg.h"
#include <atlframe.h>
#include <atlgdi.h>
#include "setmapdlg.h"
#include "findpathcode.h"
#include <atlstr.h>

#include <atlctrls.h>
enum ePathNode
{
	E_NORMAL= 0,
	E_TREE,
	E_WATER,
	E_NNN,
	E_DEF,
};

class CMainDlg : public CDialogImpl<CMainDlg>, public CUpdateUI<CMainDlg>,
		public CMessageFilter, public CIdleHandler
{
public:
	enum { IDD = IDD_MAINDLG };

	virtual BOOL PreTranslateMessage(MSG* pMsg)
	{
		return CWindow::IsDialogMessage(pMsg);
	}

	virtual BOOL OnIdle()
	{
		return FALSE;
	}

	BEGIN_UPDATE_UI_MAP(CMainDlg)
	END_UPDATE_UI_MAP()

	//WM_LBUTTONDBLCLK
	BEGIN_MSG_MAP(CMainDlg)
		MESSAGE_HANDLER(WM_INITDIALOG, OnInitDialog)
		MESSAGE_HANDLER(WM_DESTROY, OnDestroy)

		MESSAGE_HANDLER(WM_PAINT, OnPaint)
		MESSAGE_HANDLER(WM_LBUTTONDBLCLK, OnSet)

		/*
#define WM_MOUSEMOVE                    0x0200
#define WM_LBUTTONDOWN                  0x0201
#define WM_LBUTTONUP                    0x0202
		*/

		MESSAGE_HANDLER(WM_MOUSEMOVE, OnMouseMove)
		MESSAGE_HANDLER(WM_LBUTTONDOWN, OnLBtD)
		MESSAGE_HANDLER(WM_LBUTTONUP, OnLBtU)

		COMMAND_ID_HANDLER(ID_APP_ABOUT, OnAppAbout)
		COMMAND_ID_HANDLER(IDOK, OnOK)
		COMMAND_ID_HANDLER(IDCANCEL, OnCancel)
		COMMAND_HANDLER(IDC_BUTTON1, BN_CLICKED, OnBnClickedButton1)
		COMMAND_HANDLER(IDC_BUTTON2, BN_CLICKED, OnBnClickedButton2)
		COMMAND_HANDLER(IDC_BUTTON4, BN_CLICKED, OnBnClickedButton4)
		COMMAND_HANDLER(IDC_BUTTON5, BN_CLICKED, OnBnClickedButton5)
		COMMAND_HANDLER(IDC_BUTTON6, BN_CLICKED, OnBnClickedButton6)
	END_MSG_MAP()

// Handler prototypes (uncomment arguments if needed):
//	LRESULT MessageHandler(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM /*lParam*/, BOOL& /*bHandled*/)
//	LRESULT CommandHandler(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
//	LRESULT NotifyHandler(int /*idCtrl*/, LPNMHDR /*pnmh*/, BOOL& /*bHandled*/)


    int GetIndexc(int i, int j )
	{
		return i*m_nNum + j;
	}
	
	LRESULT OnMouseMove(UINT /*uMsg*/, WPARAM wParam, LPARAM lParam, BOOL& /*bHandled*/)
	{
		if( MK_LBUTTON & wParam )
		{
			 POINT p;
			 p.x = GET_X_LPARAM(lParam);
			 p.y = GET_Y_LPARAM(lParam);			 

			 int i = (p.x - 20) / m_nd; 
			 int j = (p.y - 20) / m_nd;
			 m_nnMap[GetIndexc(i,j)] = m_nmapCor;
			

			 if( p.x !=  m_pMakeMapLast.x || p.y != m_pMakeMapLast.y )
			 {
				 
			 }
			 m_pMakeMapLast = p;			 
		}


		return 0;
	}
	LRESULT OnLBtD(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM lParam, BOOL& /*bHandled*/)
	{

		m_bBtd = true;
		return 0;
	}
    LRESULT OnLBtU(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM lParam, BOOL& /*bHandled*/)
	{
		m_bBtd = false;
		InvalidateRect(NULL);

		return 0;
	}

    LRESULT OnSet(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM lParam, BOOL& /*bHandled*/)
	{
		 POINT p;
         p.x = GET_X_LPARAM(lParam);
         p.y = GET_Y_LPARAM(lParam);

		 CSetDlg dlg;
		 if( dlg.DoModal() == IDOK)
		 {
			 int i = (p.x - 20) / m_nd; 
			 int j = (p.y - 20) / m_nd; 
			 m_nnMap[GetIndexc(i,j)] = dlg.m_nIndex1;

			 if( dlg.m_nIndex2 == 1)
			 {
				 m_pStart.x = i;
				 m_pStart.y = j;
			 }
			  if( dlg.m_nIndex2 == 2)
			  {
				  m_pEnd.x = i;
			      m_pEnd.y = j;
			  }			 
			  m_bHasres = false;
			  InvalidateRect(NULL);
		 }
		 else
		 {
			 return 1;
		 }
		 return 0;
	}
    LRESULT OnPaint(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM /*lParam*/, BOOL& /*bHandled*/)
	{
		CPaintDC dc(m_hWnd);
		dc.SaveDC();
        //TODO: Add your drawing code here

		 int xs = m_nx0 + m_nd * m_nNum;
		 int ys = m_ny0 + m_nd * m_nNum;
		 
		 for(int i=0;i<=m_nNum;i++)
		 {
			 int dy = 0;
			 for( int j=0;j<=m_nNum;j++)
			 {
				   dc.MoveTo( m_nx0 , m_ny0 + dy ); 
				   dc.LineTo( xs ,    m_ny0 + dy);
				   dy += m_nd;
			 }

			 int dx = 0;
			 for( int j=0;j<=m_nNum;j++)
			 {
				   dc.MoveTo( m_nx0 + dx, m_ny0); 
				   dc.LineTo( m_nx0 + dx ,  ys );
				   dx += m_nd;
			 }


		 }
		 // 地图信息
		 {	
			 for( int i=0;i<m_nNum * m_nNum; i++)
			 {
				 if( m_nnMap[i]>0 )
				 {
					 DrawColor(dc,i/m_nNum,i%m_nNum,m_nnMap[i]);
				 }
			 }

			 if( m_pStart.x != -1 && m_pStart.y != -1 )
			 {
				 dc.TextOutW( m_nx0 + m_pStart.x * m_nd+3 ,m_ny0 + m_pStart.y * m_nd,L"s");
			 }

			 if( m_pEnd.x != -1 && m_pEnd.y != -1 )
			 {
				 dc.TextOutW( m_nx0 + m_pEnd.x * m_nd+3,m_ny0 + m_pEnd.y * m_nd,L"e");
			 }
		 }


		 // 路径信息
		 if( m_bHasres )
		 {
			 //  哪些点被分析过了
			 tagNode * pOutList  = m_pOutList1;
			 if( m_bBfs )
			 {
				 pOutList = m_pOutList1;
			 }
			 else
			 {
				 pOutList = m_pOutList2;
			 }

			 if( m_bShowWork )
			 {
				for( int i=0;i<m_nNum*m_nNum;i++)
				{
					 if( pOutList[i].bAnnalyze == true )
					 {
						 
						   DrawColor(dc,i/m_nNum,i%m_nNum,-1); // 标记被分析了的点
					 }
				 }
			 }	

			 int nCurX = m_pEnd.x;
			 int nCurY = m_pEnd.y;	

			 ATL::CString strResult;
			 strResult.Format(L"%d",pOutList[GetIndexc( nCurX,nCurY)].nGot);
			 ::SendMessage( GetDlgItem(IDC_RESULT), WM_SETTEXT, 0, (LPARAM)strResult.GetBuffer()) ;	

			 // 划线
			 while(true)
			 {
				 int nIndexC = GetIndexc( nCurX,nCurY);

				 int nFatherX= pOutList[nIndexC].pFather.x;
				 int nFatherY= pOutList[nIndexC].pFather.y;

				 if( nCurX!= -1 && nFatherX != -1 )
				 {
					  dc.MoveTo( m_nx0 + nCurX*m_nd+ m_nd/2 ,    m_ny0 + nCurY*m_nd+ m_nd/2 ); 
				      dc.LineTo( m_nx0 + nFatherX*m_nd+ m_nd/2 , m_ny0 + nFatherY*m_nd+ m_nd/2 ); 	
				 }	

				 nCurX = nFatherX; nCurY=nFatherY;
				 if( nCurX == m_pStart.x && nCurY == m_pStart.y)
				 {
					 break;
				 }				 
			 }


		 }
		 dc.RestoreDC(-1);
	     return 0;
	}

	void DrawColor( CPaintDC &dcobj,int i, int j, int eColor = 0 )
	{
		if( i==m_pStart.x && j==m_pStart.y )
		{
			return ;
		}
		if( i==m_pEnd.x && j==m_pEnd.y )
		{
			return ;
		}
		CBrush brush;
		int nColor = 0;
		if( eColor >=0 )
		{
			if( eColor > 4 )
			{
				eColor =4;
			}
			nColor = eColor * 50;
			if( eColor == E_DEF )
			{
				nColor = 255;
			}

			nColor = 255 - nColor;
			if(  nColor == 255 )
			{
				return ;
			}
			brush.CreateSolidBrush(RGB(nColor,nColor,nColor));
		}
		else
		{
			brush.CreateSolidBrush(RGB(255,0,0));
		}
	



		
		RECT rect;
		rect.left = m_nx0 + i * m_nd;
		rect.top = m_ny0 + j * m_nd;
		rect.right = m_nx0 + i * m_nd + m_nd;
		rect.bottom =m_ny0 + j * m_nd + m_nd;
		dcobj.FillRect( &rect,brush.m_hBrush);	
	 
	}

	LRESULT OnInitDialog(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM /*lParam*/, BOOL& /*bHandled*/)
	{
		// center the dialog on the screen
		CenterWindow();

		// set icons
		HICON hIcon = (HICON)::LoadImage(_Module.GetResourceInstance(), MAKEINTRESOURCE(IDR_MAINFRAME), 
			IMAGE_ICON, ::GetSystemMetrics(SM_CXICON), ::GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR);
		SetIcon(hIcon, TRUE);
		HICON hIconSmall = (HICON)::LoadImage(_Module.GetResourceInstance(), MAKEINTRESOURCE(IDR_MAINFRAME), 
			IMAGE_ICON, ::GetSystemMetrics(SM_CXSMICON), ::GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR);
		SetIcon(hIconSmall, FALSE);

		// register object for message filtering and idle updates
		CMessageLoop* pLoop = _Module.GetMessageLoop();
		ATLASSERT(pLoop != NULL);
		pLoop->AddMessageFilter(this);
		pLoop->AddIdleHandler(this);

		UIAddChildWindowContainer(m_hWnd);


		m_nNum = 36;
		m_nx0 = 20;
		m_ny0 = 20;
	    m_nd = 16;

		m_nnMap = new int[m_nNum*m_nNum];	
		for( int i=0;i<m_nNum*m_nNum;i++)
		{
			m_nnMap[i] = 0;
		}
        m_pStart.x = -1;
		m_pStart.y = -1;

		m_pEnd.x = -1;
		m_pEnd.y = -1;

		m_pOutList1 = new tagNode[m_nNum*m_nNum];
		m_pOutList2 = new tagNode[m_nNum*m_nNum];
		m_bHasres = false;
		m_bBfs    = true;
		m_bShowWork = true;
		m_bMakemapM = false;
		m_nmapCor = 0;
		m_bBtd = false;

		m_comBoRate.Attach(GetDlgItem( IDC_COMBO_RATE ) );

		m_comBoRate.InsertString(0,L"1.0");
		m_comBoRate.InsertString(1,L"1.5");
		m_comBoRate.InsertString(2,L"2.0");
		m_comBoRate.InsertString(3,L"3.0");

		m_comBoRate.SetCurSel(0);

		return TRUE;
	}

	LRESULT OnDestroy(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM /*lParam*/, BOOL& /*bHandled*/)
	{
		// unregister message filtering and idle updates
		CMessageLoop* pLoop = _Module.GetMessageLoop();
		ATLASSERT(pLoop != NULL);
		pLoop->RemoveMessageFilter(this);
		pLoop->RemoveIdleHandler(this);

		return 0;
	}

	LRESULT OnAppAbout(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		CAboutDlg dlg;
		dlg.DoModal();
		return 0;
	}

	LRESULT OnOK(WORD /*wNotifyCode*/, WORD wID, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		// TODO: Add validation code 
		CloseDialog(wID);
		return 0;
	}

	LRESULT OnCancel(WORD /*wNotifyCode*/, WORD wID, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		CloseDialog(wID);
		return 0;
	}

	void CloseDialog(int nVal)
	{
		DestroyWindow();
		::PostQuitMessage(nVal);
	}

private:
		int  m_nNum ;
		int   m_nx0 ;
		int   m_ny0 ;
		int   m_nd ;

		int   *m_nnMap;
		POINT m_pStart;
		POINT m_pEnd;
		POINT m_pMakeMapLast;

		tagNode * m_pOutList1 ;
		tagNode * m_pOutList2 ;

		bool m_bHasres;
		bool m_bBfs;
		bool m_bShowWork;
		bool m_bMakemapM;
		int  m_nmapCor;
		bool m_bBtd;

		WTL::CComboBox  m_comBoRate;
public:
	LRESULT OnBnClickedButton1(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		for( int i=0;i<m_nNum*m_nNum;i++)
		{
			m_nnMap[i] = 0;
		}

		m_bHasres = false;
		

		InvalidateRect(NULL);


		return 0;
	}
public:
	LRESULT OnBnClickedButton2(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		if( m_pStart.x == -1 || m_pEnd.x == -1)
		{
			MessageBox(L"没有起点终点",L"错误");
			return 1;
		}
        for( int i=0;i<m_nNum*m_nNum;i++)
		{
			tagNode node;
			m_pOutList1[i] = node;
			m_pOutList2[i] = node;
		}

		double lfRate = 1.0;
		int nCurSet = m_comBoRate.GetCurSel();
		if( nCurSet == 1 )
		{
			lfRate = 1.5;
		}
		if( nCurSet == 2 )
		{
			lfRate = 2.0;
		}
		if( nCurSet == 3 )
		{
			lfRate = 3.0;
		}

		int nRes = FindPathByW( m_nnMap, m_nNum, m_nNum, m_pStart, m_pEnd,m_pOutList1, true);
		if( nRes == 0 )
		{
			nRes = FindPathByW( m_nnMap, m_nNum, m_nNum, m_pStart, m_pEnd,m_pOutList2, false, lfRate);
		}

		if( nRes != 0 )
		{
			MessageBox(L"路径无法到达",L"提示");
			return 1;
		}
	    
		m_bHasres = true;

		InvalidateRect(NULL);

		return 0;
	}
	LRESULT OnBnClickedButton4(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		m_bBfs = !m_bBfs;
		
		if( m_bBfs)
		{
			::SendMessage( GetDlgItem(IDC_BUTTON4), WM_SETTEXT, 0, (LPARAM)L"这是bfs算法") ;
		}
		else
		{
			::SendMessage( GetDlgItem(IDC_BUTTON4), WM_SETTEXT, 0, (LPARAM) L"这是a*") ;
		}
		

		InvalidateRect(NULL);
		return 0;
	}
	LRESULT OnBnClickedButton5(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		m_bShowWork = !m_bShowWork;

		if( m_bShowWork)
		{
			::SendMessage( GetDlgItem(IDC_BUTTON5), WM_SETTEXT, 0, (LPARAM)L"隐藏分析的节点") ;
		}
		else
		{
			::SendMessage( GetDlgItem(IDC_BUTTON5), WM_SETTEXT, 0, (LPARAM)L"显示分析的节点") ;
		}
		
		InvalidateRect(NULL);
		return 0;
	}
    LRESULT OnBnClickedButton6(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		if( !m_bMakemapM )
		{
			 CSetDlg dlg;
			 if( dlg.DoModal() == IDOK)
			 {
				 m_nmapCor = dlg.m_nIndex1;
			 }

			 ::SendMessage( GetDlgItem(IDC_BUTTON6), WM_SETTEXT, 0, (LPARAM)L"取消制图") ;			
		}
		else
		{
			 ::SendMessage( GetDlgItem(IDC_BUTTON6), WM_SETTEXT, 0, (LPARAM)L"设置制图") ;	
		}

		m_bMakemapM = !m_bMakemapM;
		return 0;
	}
};
