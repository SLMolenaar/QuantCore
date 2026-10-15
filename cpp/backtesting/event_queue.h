#pragma once

#include "event.h"
#include <queue>
#include <vector>
#include <stdexcept>

namespace quantcore {

    //Priority queue for events ordered by timestamp
    class EventQueue {
    public:
        EventQueue() = default;

        // Add event to queue
        void push(EventPtr event) {
            if (!event) {
                throw std::invalid_argument("Cannot push null event");
            }
            queue_.push({std::move(event), next_seq_++});
        }

        EventPtr pop() {
            if (queue_.empty()) {
                throw std::runtime_error("Cannot pop from empty queue");
            }

            EventPtr event = queue_.top().ev;
            queue_.pop();
            return event;
        }

        EventPtr peek() const {
            if (queue_.empty()) {
                throw std::runtime_error("Queue is empty");
            }
            return queue_.top().ev;
        }

        bool empty() const {
            return queue_.empty();
        }

        size_t size() const {
            return queue_.size();
        }

        void clear() {
            while (!queue_.empty()) {
                queue_.pop();
            }
        }

    private:
        // Comparator for min-heap
        struct Entry {
            EventPtr ev;
            uint64_t seq;   // insertion order, makes ties deterministic (FIFO)
        };

        struct EventComparator {
            bool operator()(const Entry& l, const Entry& r) const {
                if (l.ev->get_timestamp() != r.ev->get_timestamp())
                    return l.ev->get_timestamp() > r.ev->get_timestamp();
                // Tie-break by event type: m_data < sig < order < fill
                if (l.ev->get_type() != r.ev->get_type())
                    return l.ev->get_type() > r.ev->get_type();
                return l.seq > r.seq;
            }
        };

        uint64_t next_seq_ = 0;
        std::priority_queue<Entry, std::vector<Entry>, EventComparator> queue_;
    };

}