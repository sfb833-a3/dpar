// Copyright 2015 The dpar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package system

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// A label numberer creates a bijection between (string-based)
// features and numbers.
type LabelNumberer struct {
	labelNumbers map[Transition]int
	labels       []Transition
}

func NewLabelNumberer() LabelNumberer {
	return LabelNumberer{make(map[Transition]int), make([]Transition, 0)}
}

func (l *LabelNumberer) Number(label Transition) int {
	idx, ok := l.labelNumbers[label]

	if !ok {
		idx = len(l.labelNumbers)
		l.labelNumbers[label] = idx
		l.labels = append(l.labels, label)
	}

	return idx
}

func (l LabelNumberer) Label(number int) Transition {
	return l.labels[number]
}

func (l LabelNumberer) Size() int {
	return len(l.labels)
}

func (l *LabelNumberer) Read(reader io.Reader, serializer TransitionSerializer) error {
	var labels []Transition
	bufReader := bufio.NewReader(reader)

	eof := false
	for !eof {
		line, err := bufReader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return err
			}

			eof = true
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if trans, err := serializer.DeserializeTransition(strings.TrimSpace(line)); err == nil {
			labels = append(labels, trans)
		} else {
			return err
		}
	}

	numbers := make(map[Transition]int)
	for idx, label := range labels {
		numbers[label] = idx
	}

	l.labels = labels
	l.labelNumbers = numbers

	return nil
}

func (l *LabelNumberer) WriteLabelNumberer(writer io.Writer, serializer TransitionSerializer) error {
	for _, trans := range l.labels {
		if transStr, err := serializer.SerializeTransition(trans); err == nil {
			fmt.Fprintf(writer, "%s\n", transStr)
		} else {
			return err
		}
	}

	return nil
}