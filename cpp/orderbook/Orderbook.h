#pragma once

#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <vector>
#include <optional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Order.h"
#include "OrderModify.h"
#include "Trade.h"
#include "LevelInfo.h"
#include "Types.h"
#include "OrderType.h"
#include "Constants.h"
#include "MarketDataFeed.h"
#include "ExchangeRules.h"

class Orderbook {
private:
    // OrderPointers is now std::vector<OrderPointer> (see Types.h).
    // location_ is an index into the vector at the order's price level.
    // When an order is cancelled, the vector is erased at that index and
    // all subsequent orders at the same level have their indices decremented.
    // In practice each price level has 1 order (market maker), so the shift
    // cost is always zero.
    struct OrderEntry {
        OrderPointer order_{nullptr};
        size_t location_; // index into OrderPointers vector at this price level
    };

    std::map<Price, OrderPointers, std::greater<Price> > bids_;
    std::map<Price, OrderPointers, std::less<Price> > asks_;
    std::unordered_map<OrderId, OrderEntry> orders_;

    std::chrono::system_clock::time_point lastDayReset_;
    std::chrono::hours dayResetHour_{15};
    int dayResetMinute_{59};

    MarketDataStats stats_;
    uint64_t lastSequenceNumber_ = 0;
    bool isInitialized_ = false;

    ExchangeRules exchangeRules_;

    bool CanMatch(Side side, Price price) const {
        if (side == Side::Buy) {
            if (asks_.empty()) return false;
            const auto &[bestAsk, _] = *asks_.begin();
            return price >= bestAsk;
        } else {
            if (bids_.empty()) return false;
            const auto &[bestBid, _] = *bids_.begin();
            return price <= bestBid;
        }
    }

    OrderValidation ValidateOrder(OrderPointer order) const {
        if (orders_.contains(order->GetOrderId())) {
            return OrderValidation::Reject(RejectReason::DuplicateOrderId);
        }

        Price orderPrice = order->GetPrice();
        bool isConvertedMarketOrder = (orderPrice == std::numeric_limits<Price>::max() ||
                                       orderPrice == std::numeric_limits<Price>::min());

        if (!isConvertedMarketOrder) {
            if (!exchangeRules_.IsValidPrice(orderPrice)) {
                return OrderValidation::Reject(RejectReason::InvalidPrice);
            }
        }

        if (!exchangeRules_.IsValidQuantity(order->GetRemainingQuantity())) {
            if (order->GetRemainingQuantity() < exchangeRules_.minQuantity) {
                return OrderValidation::Reject(RejectReason::BelowMinQuantity);
            } else if (order->GetRemainingQuantity() > exchangeRules_.maxQuantity) {
                return OrderValidation::Reject(RejectReason::AboveMaxQuantity);
            } else {
                return OrderValidation::Reject(RejectReason::InvalidQuantity);
            }
        }

        if (!isConvertedMarketOrder) {
            if (!exchangeRules_.IsValidNotional(order->GetPrice(), order->GetRemainingQuantity())) {
                return OrderValidation::Reject(RejectReason::BelowMinNotional);
            }
        }

        return OrderValidation::Accept();
    }

    void CheckAndResetDay() {
        auto now = std::chrono::system_clock::now();
        auto nowTime = std::chrono::system_clock::to_time_t(now);
        auto lastResetTime = std::chrono::system_clock::to_time_t(lastDayReset_);

        std::tm nowTm = *std::localtime(&nowTime);

        std::tm todayResetTm = nowTm;
        todayResetTm.tm_hour = dayResetHour_.count();
        todayResetTm.tm_min = dayResetMinute_;
        todayResetTm.tm_sec = 0;
        auto todayResetTime = std::mktime(&todayResetTm);

        if (lastResetTime < todayResetTime && nowTime >= todayResetTime) {
            CancelGoodForDayOrders();
            lastDayReset_ = now;
        }
    }

    void CancelGoodForDayOrders() {
        std::vector<OrderId> ordersToCancel;
        for (const auto &[orderId, entry]: orders_) {
            if (entry.order_->GetOrderType() == OrderType::GoodForDay) {
                ordersToCancel.push_back(orderId);
            }
        }
        for (const auto &orderId: ordersToCancel) {
            CancelOrder(orderId);
        }
    }

    std::vector<std::pair<OrderPointer, Quantity> > CollectMatchesForFillOrKill(
        OrderPointer order,
        Quantity &remainingQuantity) {

        std::vector<std::pair<OrderPointer, Quantity> > matchingOrders;

        if (order->GetSide() == Side::Buy) {
            for (auto &[askPrice, askOrders]: asks_) {
                if (askPrice > order->GetPrice()) break;
                for (auto &ask: askOrders) {
                    Quantity matchQty = std::min(remainingQuantity, ask->GetRemainingQuantity());
                    matchingOrders.push_back({ask, matchQty});
                    remainingQuantity -= matchQty;
                    if (remainingQuantity == 0) break;
                }
                if (remainingQuantity == 0) break;
            }
        } else {
            for (auto &[bidPrice, bidOrders]: bids_) {
                if (bidPrice < order->GetPrice()) break;
                for (auto &bid: bidOrders) {
                    Quantity matchQty = std::min(remainingQuantity, bid->GetRemainingQuantity());
                    matchingOrders.push_back({bid, matchQty});
                    remainingQuantity -= matchQty;
                    if (remainingQuantity == 0) break;
                }
                if (remainingQuantity == 0) break;
            }
        }

        return matchingOrders;
    }

    Trades ExecuteMatchesForFillOrKill(
        OrderPointer order,
        const std::vector<std::pair<OrderPointer, Quantity> > &matchingOrders) {
        Trades trades;

        for (auto &[matchOrder, quantity]: matchingOrders) {
            Price tradePrice = matchOrder->GetPrice();
            order->Fill(quantity);
            matchOrder->Fill(quantity);

            if (order->GetSide() == Side::Buy) {
                trades.push_back(Trade{
                    TradeInfo{order->GetOrderId(), tradePrice, quantity},
                    TradeInfo{matchOrder->GetOrderId(), tradePrice, quantity}
                });
            } else {
                trades.push_back(Trade{
                    TradeInfo{matchOrder->GetOrderId(), tradePrice, quantity},
                    TradeInfo{order->GetOrderId(), tradePrice, quantity}
                });
            }

            if (matchOrder->IsFilled()) {
                CancelOrder(matchOrder->GetOrderId());
            }
        }

        return trades;
    }

    Trades MatchFillOrKill(OrderPointer order) {
        Quantity remainingQuantity = order->GetRemainingQuantity();
        auto matchingOrders = CollectMatchesForFillOrKill(order, remainingQuantity);
        if (remainingQuantity > 0) return {};
        return ExecuteMatchesForFillOrKill(order, matchingOrders);
    }

    Trades MatchOrders() {
        // No upfront reserve - see change 1 comment.
        Trades trades;

        while (true) {
            if (bids_.empty() || asks_.empty()) break;

            auto &[bidPrice, bids] = *bids_.begin();
            auto &[askPrice, asks] = *asks_.begin();

            if (bidPrice < askPrice) break;

            std::vector<OrderId> filledOrders;

            // Track how many orders we consume from the front of each level.
            // Using indices instead of pop_front() avoids O(n) shifting per
            // consumed order. We erase the consumed prefix in one shot after
            // the inner loop, then fix up stored indices for remaining orders.
            size_t bid_consumed = 0;
            size_t ask_consumed = 0;

            while (bid_consumed < bids.size() && ask_consumed < asks.size()) {
                auto &bid = bids[bid_consumed];
                auto &ask = asks[ask_consumed];

                Quantity quantity = std::min(bid->GetRemainingQuantity(), ask->GetRemainingQuantity());

                Price tradePrice;
                bool bidIsMarket = (bid->GetPrice() == std::numeric_limits<Price>::max() ||
                                    bid->GetPrice() == std::numeric_limits<Price>::min());
                bool askIsMarket = (ask->GetPrice() == std::numeric_limits<Price>::max() ||
                                    ask->GetPrice() == std::numeric_limits<Price>::min());

                if (bidIsMarket && !askIsMarket)       tradePrice = ask->GetPrice();
                else if (askIsMarket && !bidIsMarket)  tradePrice = bid->GetPrice();
                else                                    tradePrice = ask->GetPrice();

                trades.push_back(Trade{
                    TradeInfo{bid->GetOrderId(), tradePrice, quantity},
                    TradeInfo{ask->GetOrderId(), tradePrice, quantity}
                });

                bid->Fill(quantity);
                ask->Fill(quantity);

                if (bid->IsFilled()) {
                    filledOrders.push_back(bid->GetOrderId());
                    bid_consumed++;
                }
                if (ask->IsFilled()) {
                    filledOrders.push_back(ask->GetOrderId());
                    ask_consumed++;
                }
            }

            // Remove consumed orders from lookup map
            for (const auto &orderId: filledOrders) {
                orders_.erase(orderId);
            }

            // Erase consumed prefix from bid level and fix up stored indices
            if (bid_consumed > 0) {
                bids.erase(bids.begin(), bids.begin() + bid_consumed);
                for (size_t i = 0; i < bids.size(); ++i) {
                    orders_.at(bids[i]->GetOrderId()).location_ = i;
                }
            }

            // Erase consumed prefix from ask level and fix up stored indices
            if (ask_consumed > 0) {
                asks.erase(asks.begin(), asks.begin() + ask_consumed);
                for (size_t i = 0; i < asks.size(); ++i) {
                    orders_.at(asks[i]->GetOrderId()).location_ = i;
                }
            }

            if (bids.empty()) bids_.erase(bidPrice);
            if (asks.empty()) asks_.erase(askPrice);
        }

        // Handle IOC orders
        std::vector<OrderId> iocOrdersToCancel;
        for (const auto &[orderId, entry]: orders_) {
            if (entry.order_->GetOrderType() == OrderType::ImmediateOrCancel) {
                iocOrdersToCancel.push_back(orderId);
            }
        }
        for (const auto &orderId: iocOrdersToCancel) {
            CancelOrder(orderId);
        }

        return trades;
    }

    void ProcessNewOrder(const NewOrderMessage &msg) {
        try {
            auto order = std::make_shared<Order>(
                msg.orderType, msg.orderId, msg.side, msg.price, msg.quantity
            );
            auto trades = AddOrder(order);
            stats_.newOrders++;
            stats_.trades += trades.size();
        } catch (const std::invalid_argument &) {
            stats_.errors++;
        }
    }

    void ProcessCancel(const CancelOrderMessage &msg) {
        CancelOrder(msg.orderId);
        stats_.cancellations++;
    }

    void ProcessModify(const ModifyOrderMessage &msg) {
        OrderModify modify(msg.orderId, msg.side, msg.newPrice, msg.newQuantity);
        MatchOrder(modify);
        stats_.modifications++;
    }

    void ProcessTrade(const TradeMessage &msg) {
        stats_.trades++;
    }

    void ProcessSnapshot(const BookSnapshotMessage &msg) {
        bids_.clear();
        asks_.clear();
        orders_.clear();

        OrderId syntheticId = 0x8000000000000000ULL;

        for (const auto &level: msg.bids) {
            if (level.quantity == 0) continue;
            try {
                auto order = std::make_shared<Order>(
                    OrderType::GoodTillCancel, syntheticId++,
                    Side::Buy, level.price, level.quantity
                );
                auto &orders = bids_[level.price];
                orders.push_back(order);
                size_t idx = orders.size() - 1;
                orders_.insert({order->GetOrderId(), OrderEntry{order, idx}});
            } catch (const std::invalid_argument &) { continue; }
        }

        for (const auto &level: msg.asks) {
            if (level.quantity == 0) continue;
            try {
                auto order = std::make_shared<Order>(
                    OrderType::GoodTillCancel, syntheticId++,
                    Side::Sell, level.price, level.quantity
                );
                auto &orders = asks_[level.price];
                orders.push_back(order);
                size_t idx = orders.size() - 1;
                orders_.insert({order->GetOrderId(), OrderEntry{order, idx}});
            } catch (const std::invalid_argument &) { continue; }
        }

        isInitialized_ = true;
        lastSequenceNumber_ = msg.sequenceNumber;
        stats_.snapshots++;
    }

public:
    Orderbook()
        : lastDayReset_(std::chrono::system_clock::now()) {
    }

    void SetExchangeRules(const ExchangeRules &rules) {
        exchangeRules_ = rules;
    }

    const ExchangeRules &GetExchangeRules() const {
        return exchangeRules_;
    }

    void SetDayResetTime(int hour, int minute = 59) {
        if (hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59) {
            dayResetHour_ = std::chrono::hours(hour);
            dayResetMinute_ = minute;
        }
    }

    // CheckAndResetDay() removed from hot path.
    Trades AddOrder(OrderPointer order) {
        if (order->GetOrderType() == OrderType::Market) {
            if (order->GetSide() == Side::Buy && !asks_.empty()) {
                order->ToGoodTillCancel(std::numeric_limits<Price>::max());
            } else if (order->GetSide() == Side::Sell && !bids_.empty()) {
                order->ToGoodTillCancel(std::numeric_limits<Price>::min());
            } else {
                return {};
            }
        }

        auto validation = ValidateOrder(order);
        if (!validation.isValid) return {};

        if (order->GetOrderType() == OrderType::ImmediateOrCancel &&
            !CanMatch(order->GetSide(), order->GetPrice())) {
            return {};
        }

        if (order->GetOrderType() == OrderType::FillOrKill) {
            return MatchFillOrKill(order);
        }

        // push_back then store size-1 as index: O(1), no iterator scan needed.
        // With std::list, std::next(orders.begin(), orders.size()-1) was O(n).
        if (order->GetSide() == Side::Buy) {
            auto &orders = bids_[order->GetPrice()];
            orders.push_back(order);
            size_t idx = orders.size() - 1;
            orders_.insert({order->GetOrderId(), OrderEntry{order, idx}});
        } else {
            auto &orders = asks_[order->GetPrice()];
            orders.push_back(order);
            size_t idx = orders.size() - 1;
            orders_.insert({order->GetOrderId(), OrderEntry{order, idx}});
        }

        return MatchOrders();
    }

    void CancelOrder(OrderId orderId) {
        auto it = orders_.find(orderId);
        if (it == orders_.end()) return;

        const auto &[order, idx] = it->second;
        orders_.erase(it);

        auto &level = (order->GetSide() == Side::Sell)
            ? asks_.at(order->GetPrice())
            : bids_.at(order->GetPrice());

        // Erase at index. Then fix up stored indices for all orders that
        // shifted left. With 1 order per level (typical MM usage), this
        // loop body never executes.
        level.erase(level.begin() + idx);
        for (size_t i = idx; i < level.size(); ++i) {
            orders_.at(level[i]->GetOrderId()).location_ = i;
        }

        if (level.empty()) {
            if (order->GetSide() == Side::Sell) asks_.erase(order->GetPrice());
            else bids_.erase(order->GetPrice());
        }
    }

    // CheckAndResetDay() removed from hot path.
    Trades MatchOrder(OrderModify order) {
        if (!orders_.contains(order.GetOrderId())) return {};
        const auto &[existingOrder, _] = orders_.at(order.GetOrderId());
        CancelOrder(order.GetOrderId());
        return AddOrder(order.ToOrderPointer(existingOrder->GetOrderType()));
    }

    std::size_t Size() const { return orders_.size(); }

    OrderbookLevelInfos GetOrderInfos() const {
        LevelInfos bidInfos, askInfos;
        bidInfos.reserve(orders_.size());
        askInfos.reserve(orders_.size());

        auto CreateLevelInfos = [](Price price, const OrderPointers &orders) {
            return LevelInfo{
                price, std::accumulate(orders.begin(), orders.end(), (Quantity)0,
                    [](std::size_t runningSum, const OrderPointer &order) {
                        return runningSum + order->GetRemainingQuantity();
                    })
            };
        };

        for (const auto &[price, orders]: bids_) {
            bidInfos.push_back(CreateLevelInfos(price, orders));
        }
        for (const auto &[price, orders]: asks_) {
            askInfos.push_back(CreateLevelInfos(price, orders));
        }

        return OrderbookLevelInfos(bidInfos, askInfos);
    }

    bool ProcessMarketData(const MarketDataMessage &message) {
        auto startTime = std::chrono::high_resolution_clock::now();
        try {
            std::visit([this](auto &&msg) {
                using T = std::decay_t<decltype(msg)>;
                if constexpr (std::is_same_v<T, NewOrderMessage>)        ProcessNewOrder(msg);
                else if constexpr (std::is_same_v<T, CancelOrderMessage>) ProcessCancel(msg);
                else if constexpr (std::is_same_v<T, ModifyOrderMessage>) ProcessModify(msg);
                else if constexpr (std::is_same_v<T, TradeMessage>)       ProcessTrade(msg);
                else if constexpr (std::is_same_v<T, BookSnapshotMessage>) ProcessSnapshot(msg);
            }, message);

            auto endTime = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
            stats_.messagesProcessed++;
            stats_.totalProcessingTime += latency;
            stats_.maxLatency = std::max(stats_.maxLatency, latency);
            stats_.minLatency = std::min(stats_.minLatency, latency);
            return true;
        } catch (...) {
            stats_.errors++;
            return false;
        }
    }

    size_t ProcessMarketDataBatch(const std::vector<MarketDataMessage> &messages) {
        size_t successCount = 0;
        for (const auto &msg: messages) {
            if (ProcessMarketData(msg)) successCount++;
        }
        return successCount;
    }

    const MarketDataStats &GetMarketDataStats() const { return stats_; }
    void ResetMarketDataStats() { stats_.Reset(); }
    bool IsInitialized() const { return isInitialized_; }
    uint64_t GetLastSequenceNumber() const { return lastSequenceNumber_; }
};