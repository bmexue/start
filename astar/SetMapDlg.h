#pragma once

#include "resource.h"
#include "aboutdlg.h"
#include <atlframe.h>
#include <atlgdi.h>
#include "atlctrls.h"

// CSetMapDlg dialog

class CSetDlg : public CDialogImpl<CSetDlg>, public CUpdateUI<CSetDlg>,
		public CMessageFilter, public CIdleHandler
{
public:
	enum { IDD = IDD_SET_DLG };

	virtual BOOL PreTranslateMessage(MSG* pMsg)
	{
		return CWindow::IsDialogMessage(pMsg);
	}

	virtual BOOL OnIdle()
	{
		return FALSE;
	}

	BEGIN_UPDATE_UI_MAP(CSetDlg)
	END_UPDATE_UI_MAP()

	//WM_LBUTTONDBLCLK
	BEGIN_MSG_MAP(CSetDlg)
		MESSAGE_HANDLER(WM_INITDIALOG, OnInitDialog)
		MESSAGE_HANDLER(WM_DESTROY, OnDestroy)

		COMMAND_ID_HANDLER(IDOK, OnOK)
		COMMAND_ID_HANDLER(IDCANCEL, OnCancel)
	END_MSG_MAP()

// Handler prototypes (uncomment arguments if needed):
//	LRESULT MessageHandler(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM /*lParam*/, BOOL& /*bHandled*/)
//	LRESULT CommandHandler(WORD /*wNotifyCode*/, WORD /*wID*/, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
//	LRESULT NotifyHandler(int /*idCtrl*/, LPNMHDR /*pnmh*/, BOOL& /*bHandled*/)


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


		m_boxObj.Attach(GetDlgItem( IDC_COMBO1 ) );
		m_boxObj.InsertString(0,L"普通");
		m_boxObj.InsertString(1,L"树林");
		m_boxObj.InsertString(2,L"河流");
		m_boxObj.InsertString(3,L"沼泽");
		m_boxObj.InsertString(4,L"障碍");
		m_boxObj.SetCurSel(0);

		m_boxObj2.Attach(GetDlgItem( IDC_COMBO2 ) );
        m_boxObj2.InsertString(0,L"普通");
		m_boxObj2.InsertString(1,L"起点");
		m_boxObj2.InsertString(2,L"终点");
		m_boxObj2.SetCurSel(0);

		return TRUE;
	}

	LRESULT OnDestroy(UINT /*uMsg*/, WPARAM /*wParam*/, LPARAM /*lParam*/, BOOL& /*bHandled*/)
	{
		// unregister message filtering and idle updates
		CMessageLoop* pLoop = _Module.GetMessageLoop();
		ATLASSERT(pLoop != NULL);
		pLoop->RemoveMessageFilter(this);
		pLoop->RemoveIdleHandler(this);

		m_boxObj2.Detach();
		m_boxObj.Detach();

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

		m_nIndex1 = m_boxObj.GetCurSel();
		m_nIndex2 = m_boxObj2.GetCurSel();

		
		return 0;
	}

	LRESULT OnCancel(WORD /*wNotifyCode*/, WORD wID, HWND /*hWndCtl*/, BOOL& /*bHandled*/)
	{
		CloseDialog(wID);
		return 0;
	}

	void CloseDialog(int nVal)
	{
		//DestroyWindow();
		EndDialog(nVal);
	}


	int m_nIndex1;
	int m_nIndex2;
private:
	

	CComboBox m_boxObj;
	CComboBox m_boxObj2;

};
