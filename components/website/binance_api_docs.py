class BinanceApiDocs:
    @classmethod
    def get_all_endpoints(cls):
        def recursive_attributes(subclass):
            attrs = {}
            for name, obj in subclass.__dict__.items():
                if name in ["get_all_endpoints"]:
                    continue
                if not name.startswith('__') and not callable(obj):
                    attrs[name] = obj
                elif isinstance(obj, type):  # Check if the object is a class
                    attrs[name] = recursive_attributes(obj)
            return attrs
        
        return recursive_attributes(cls)
    class PortfolioMargin:
        class Trade:
            NEW_UM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/New-UM-Order"
            NEW_UM_CONDITIONAL_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/New-UM-Conditional-Order"
            NEW_CM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/New-CM-Order"
            NEW_CM_CONDITIONAL_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/New-CM-Conditional-Order"
            NEW_MARGIN_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/New-Margin-Order"
            MARGIN_ACCOUNT_BORROW = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Margin-Account-Borrow"
            MARGIN_ACCOUNT_REPAY = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Margin-Account-Repay"
            MARGIN_ACCOUNT_NEW_OCO = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Margin-Account-New-OCO"
            CANCEL_UM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-UM-Order"
            CANCEL_ALL_UM_OPEN_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-All-UM-Open-Orders"
            CACNEL_ALL_UM_CONDITIONAL_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-All-UM-Conditional-Orders"
            CANCEL_CM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-CM-Order"
            CANCEL_ALL_CM_OPEN_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-All-CM-Open-Orders"
            CANCEL_ALL_CM_CONDITIONAL_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-All-CM-Conditional-Orders"
            CANCEL_MARGIN_ACCOUNT_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-Margin-Account-Order"
            CANCEL_MARGIN_ACCOUNT_OCO_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-Margin-Account-OCO-Orders"
            CANCEL_MARGIN_ACCOUNT_ALL_OPEN_ORDERS_ON_A_SYMBOL = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Cancel-Margin-Account-All-Open-Orders-on-a-Symbol"
            MODIFY_UM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Modify-UM-Order"
            MODIFY_CM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Modify-CM-Order"
            QUERY_UM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-UM-Order"
            QUERY_ALL_UM_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-All-UM-Orders"
            QUERY_CURRENT_UM_OPEN = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Current-UM-Open"
            QUERY_CURRENT_UM_OPEN_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Current-UM-Open-Order"
            QUERY_ALL_CURRENT_UM_OPEN_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-All-Current-UM-Open-Orders"
            QUERY_UM_CONDITIONAL_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-UM-Conditional-Order"
            QUERY_UM_CONDITIONAL_ORDER_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-UM-Conditional-Order-History"
            QUERY_CM_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-CM-Order"
            QUERY_ALL_CM_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-All-CM-Orders"
            QUERY_CURRENT_CM_OPEN_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Current-CM-Open-Orders"
            QUERY_ALL_CURRENT_CM_OPEN_CONDITIONAL_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-All-Current-CM-Open-Conditional-Orders"
            QUERY_CURRENT_CM_OPEN_CONDITIONAL_ORDER = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Current-CM-Open-Conditional-Order"
            QUERY_ALL_CM_CONDITIONAL_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-All-CM-Conditional-Orders"
            QUERY_CM_CONDITIONAL_ORDER_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-CM-Conditional-Order-History"
            QUERY_USERS_UM_FORCE_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Users-UM-Force-Orders"
            QUERY_USERS_CM_FORCE_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Users-CM-Force-Orders"
            QUERY_UM_MODIFY_ORDER_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-UM-Modify-Order-History"
            QUERY_CM_MODIFY_ORDER_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-CM-Modify-Order-History"
            QUERY_USERS_MARGIN_FORCE_ORDERS = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Users-Margin-Force-Orders"
            UM_ACCOUNT_TRADE_LIST = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/UM-Account-Trade-List"
            CM_ACCOUNT_TRADE_LIST = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/CM-Account-Trade-List"
            UM_POSITION_ADL_QUANTILE_ESTIMATION = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/UM-Position-ADL-Quantile-Estimation"
            CM_POSITION_ADL_QUANTILE_ESTIMATION = "https://developers.binance.com/docs/derivatives/portfolio-margin/trade/CM-Position-ADL-Quantile-Estimation"

        class Account:
            ACCOUNT_BALANCE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Account-Balance"
            ACCOUNT_INFORMATION = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Account-Information"
            MARGIN_MAX_BORROW = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Margin-Max-Borrow"
            QUERY_MARGIN_MAX_WITHDRAW = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-Margin-Max-Withdraw"
            QUERY_UM_POSITION_INFORMATION = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-UM-Position-Information"
            QUERY_CM_POSITION_INFORMATION = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-CM-Position-Information"
            CHANGE_UM_INITIAL_LEVERAGE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Change-UM-Initial-Leverage"
            CHANGE_CM_INITIAL_LEVERAGE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Change-CM-Initial-Leverage"
            CHANGE_UM_POSITION_MODE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Change-UM-Position-Mode"
            CHANGE_CM_POSITION_MODE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Change-CM-Position-Mode"
            GET_UM_CURRENT_POSITION_MODE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-UM-Current-Position-Mode"
            GET_CM_CURRENT_POSITION_MODE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-CM-Current-Position-Mode"
            UM_NOTIONAL_AND_LERVERAGE_BRACKETS = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/UM-Notional-and-Leverage-Brackets"
            CM_NOTIONAL_AND_LERVERAGE_BRACKETS = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/CM-Notional-and-Leverage-Brackets"
            PORTFOLIO_MARGIN_UM_TRADING_QUANTITATIVE_RULES_INDICATORS = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Portfolio-Margin-UM-Trading-Quantitative-Rules-Indicators"
            GET_USER_COMMISSION_RATE_FOR_UM = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-User-Commission-Rate-for-UM"
            GET_USER_COMMISSION_RATE_FOR_CM = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-User-Commission-Rate-for-CM"
            QUERY_MARGIN_LOAN_RECORD = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-Margin-Loan-Record"
            QUERY_MARGIN_REPAY_RECORD = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-Margin-Repay-Record"
            GET_AUTO_REPAY_FUTURES_STATUS = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-Auto-Repay-Futures-Status"
            CHANGE_AUTO_REPAY_FUTURES_STATUS = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Change-Auto-Repay-Futures-Status"
            GET_MARGIN_BORROWLOAN_INTEREST_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-Margin-BorrowLoan-Interest-History"
            REPAY_FUTURES_NEGATIVE_BALANCE = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Repay-Futures-Negative-Balance"
            QUERY_PORTFOLIO_MARGIN_NEGATIVE_BALANCE_INTEREST_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-Portfolio-Margin-Negative-Balance-Interest-History"
            FUND_AUTO_COLLECTION = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Fund-Auto-Collection"
            FUND_COLLECTION_BY_ASSET = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Fund-Collection-by-Asset"
            BNB_TRANSFER = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/BNB-Transfer"
            GET_UM_INCOME_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-UM-Income-History"
            GET_CM_INCOME_HISTORY = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-CM-Income-History"
            GET_UM_ACCOUNT_DETAIL = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-UM-Account-Detail"
            GET_CM_ACCOUNT_DETAIL = "https://developers.binance.com/docs/derivatives/portfolio-margin/account/Get-CM-Account-Detail"

        class GeneralInfo:
            GENERAL_INFO = "https://developers.binance.com/docs/derivatives/portfolio-margin/general-info"

        class CommonDefinition:
            COMMON_DEFINITION = "https://developers.binance.com/docs/derivatives/portfolio-margin/common-definition"

        class ErrorCode:
            ERROR_CODE = "https://developers.binance.com/docs/derivatives/portfolio-margin/error-code"


