#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <chrono>

namespace quantcore {

/**
 * All event types in the system
 */
enum class EventType {
    MARKET_DATA,  // Price/volume updates
    SIGNAL,       // Strategy signals
    ORDER,        // Order commands (new/cancel/modify)
    FILL          // Execution confirmations
};

// base class for events
class Event {
public:
    Event(EventType type, int64_t timestamp_ns)
        : type_(type)
        , timestamp_ns_(timestamp_ns)
    {
    }

    virtual ~Event() = default;

    EventType get_type() const { return type_; }

    // nanoseconds since epoch
    int64_t get_timestamp() const { return timestamp_ns_; }

    // to string for debugging
    virtual std::string to_string() const {
        std::string type_str;
        switch (type_) {
            case EventType::MARKET_DATA: type_str = "MARKET_DATA"; break;
            case EventType::SIGNAL: type_str = "SIGNAL"; break;
            case EventType::ORDER: type_str = "ORDER"; break;
            case EventType::FILL: type_str = "FILL"; break;
        }
        return "Event(type=" + type_str + ", timestamp=" + std::to_string(timestamp_ns_) + ")";
    }

protected:
    EventType type_;
    int64_t timestamp_ns_;
};

using EventPtr = std::shared_ptr<Event>;

// Comparison operators for event ordering
inline bool operator<(const Event& lhs, const Event& rhs) {
    return lhs.get_timestamp() > rhs.get_timestamp(); // Reverse for min-heap
}

inline bool operator>(const Event& lhs, const Event& rhs) {
    return lhs.get_timestamp() < rhs.get_timestamp();
}

}