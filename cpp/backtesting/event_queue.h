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
            queue_.push(event);
        }

        EventPtr pop() {
            if (queue_.empty()) {
                throw std::runtime_error("Cannot pop from empty queue");
            }

            EventPtr event = queue_.top();
            queue_.pop();
            return event;
        }

        EventPtr peek() const {
            if (queue_.empty()) {
                throw std::runtime_error("Queue is empty");
            }
            return queue_.top();
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
        struct EventComparator {
            bool operator()(const EventPtr& lhs, const EventPtr& rhs) const {
                if (lhs->get_timestamp() != rhs->get_timestamp()) {
                    return lhs->get_timestamp() > rhs->get_timestamp();
                }
                // Tie-break by event type: m_data < sig < order < fill
                return lhs->get_type() > rhs->get_type();
            }
        };

        std::priority_queue<
            EventPtr,
            std::vector<EventPtr>,
            EventComparator
        > queue_;
    };

}